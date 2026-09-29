"""
ParkVision AI inference engine (ONNX Runtime only - light enough for Streamlit Cloud).

Pipeline
  1. YOLOv8n (ONNX) finds every parking slot in the frame and gives a first empty/occupied guess.
  2. MobileNetV2 (ONNX) re-checks every slot crop (224x224) -> second opinion.
  3. Probabilities are fused (hybrid mode) -> final slot status.
  4. Insight layer: counts, occupancy %, congestion level, zone analysis, recommendation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

MODELS = Path(__file__).resolve().parents[1] / "models"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NEON_GREEN = (20, 255, 57)     # BGR
NEON_RED = (96, 56, 255)
NEON_CYAN = (255, 240, 0)


def _session(name: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    return ort.InferenceSession(str(MODELS / name), so, providers=["CPUExecutionProvider"])


@dataclass
class Slot:
    id: int
    box: tuple  # x1, y1, x2, y2 in original image pixels
    p_occ_det: float
    p_occ_cls: float | None = None
    p_occ: float = 0.0
    occupied: bool = False
    zone: str = ""


@dataclass
class ScanResult:
    slots: list = field(default_factory=list)
    total: int = 0
    occupied: int = 0
    available: int = 0
    occupancy: float = 0.0
    congestion: str = "LOW"
    recommendation: str = ""
    advice: str = ""
    zones: dict = field(default_factory=dict)
    best_zone: str | None = None
    annotated: np.ndarray | None = None


class ParkVision:
    def __init__(self):
        self.det = _session("yolo_slots.onnx")
        self.cls = _session("slot_classifier.onnx")
        self.det_in = self.det.get_inputs()[0].name
        self.cls_in = self.cls.get_inputs()[0].name

    # ---------------------------------------------------------------- detection
    def detect(self, bgr: np.ndarray, conf: float = 0.25, iou: float = 0.45):
        H, W = bgr.shape[:2]
        # PKLot frames were stretched to 640x640 during labelling, so we stretch the same way
        x = cv2.resize(bgr, (640, 640))[:, :, ::-1].astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        out = self.det.run(None, {self.det_in: x})[0][0]          # (6, 8400)
        out = out.T                                               # (8400, 6): cx, cy, w, h, s_empty, s_occ
        scores = out[:, 4:6]
        best = scores.max(1)
        keep = best > conf
        out, scores, best = out[keep], scores[keep], best[keep]
        if not len(out):
            return []
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        sx, sy = W / 640.0, H / 640.0
        boxes = np.stack([(cx - w / 2) * sx, (cy - h / 2) * sy, w * sx, h * sy], 1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), best.tolist(), conf, iou)
        idx = np.array(idx).reshape(-1)
        dets = []
        for i in idx:
            bx, by, bw, bh = boxes[i]
            s_e, s_o = scores[i]
            p_occ = float(s_o / (s_e + s_o + 1e-9))
            x1, y1 = int(max(0, bx)), int(max(0, by))
            x2, y2 = int(min(W - 1, bx + bw)), int(min(H - 1, by + bh))
            if x2 - x1 > 3 and y2 - y1 > 3:
                dets.append(((x1, y1, x2, y2), p_occ))
        return dets

    # ---------------------------------------------------------------- classification
    def classify_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """Return P(occupied) for each BGR crop."""
        if not crops:
            return np.zeros(0)
        batch = []
        for c in crops:
            c = cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
            batch.append(((c - MEAN) / STD).transpose(2, 0, 1))
        probs = []
        for i in range(0, len(batch), 64):
            logits = self.cls.run(None, {self.cls_in: np.stack(batch[i:i + 64]).astype(np.float32)})[0]
            e = np.exp(logits - logits.max(1, keepdims=True))
            probs.append((e / e.sum(1, keepdims=True))[:, 1])
        return np.concatenate(probs)

    @staticmethod
    def _crop(bgr, box, pad=0.08):
        x1, y1, x2, y2 = box
        H, W = bgr.shape[:2]
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        return bgr[int(max(0, y1 - py)):int(min(H, y2 + py)), int(max(0, x1 - px)):int(min(W, x2 + px))]

    # ---------------------------------------------------------------- full scan
    def scan(self, bgr: np.ndarray, conf=0.25, mode="hybrid", w_cls=0.5, n_zones=3,
             show_ids=True) -> ScanResult:
        dets = self.detect(bgr, conf)
        # reading order: top-to-bottom rows, then left-to-right
        dets.sort(key=lambda d: (round(((d[0][1] + d[0][3]) / 2) / max(1, bgr.shape[0] / 12)), d[0][0]))
        slots = [Slot(i + 1, box, p) for i, (box, p) in enumerate(dets)]
        if mode in ("hybrid", "classifier") and slots:
            pc = self.classify_crops([self._crop(bgr, s.box) for s in slots])
            for s, p in zip(slots, pc):
                s.p_occ_cls = float(p)
        W = bgr.shape[1]
        zone_names = ["WEST", "CENTRAL", "EAST"] if n_zones == 3 else [f"Z{i+1}" for i in range(n_zones)]
        for s in slots:
            if mode == "detector" or s.p_occ_cls is None:
                s.p_occ = s.p_occ_det
            elif mode == "classifier":
                s.p_occ = s.p_occ_cls
            else:
                s.p_occ = (1 - w_cls) * s.p_occ_det + w_cls * s.p_occ_cls
            s.occupied = s.p_occ >= 0.5
            cx = (s.box[0] + s.box[2]) / 2
            s.zone = zone_names[min(n_zones - 1, int(cx / W * n_zones))]
        r = insights(slots, zone_names)
        r.annotated = draw(bgr, slots, show_ids)
        return r


# -------------------------------------------------------------------- insight logic
def congestion_level(occ_pct: float) -> str:
    if occ_pct < 40:
        return "LOW"
    if occ_pct <= 75:
        return "MODERATE"
    return "HIGH"


def insights(slots: list[Slot], zone_names: list[str]) -> ScanResult:
    r = ScanResult(slots=slots)
    r.total = len(slots)
    r.occupied = sum(s.occupied for s in slots)
    r.available = r.total - r.occupied
    r.occupancy = 100.0 * r.occupied / r.total if r.total else 0.0
    r.congestion = congestion_level(r.occupancy)
    for z in zone_names:
        zs = [s for s in slots if s.zone == z]
        free = sum(not s.occupied for s in zs)
        r.zones[z] = {"total": len(zs), "free": free, "occupancy": 100.0 * (len(zs) - free) / len(zs) if zs else 0.0}
    cand = [(v["free"], z) for z, v in r.zones.items() if v["total"]]
    r.best_zone = max(cand)[1] if cand and max(cand)[0] > 0 else None

    if r.total == 0:
        r.recommendation, r.advice = "NO SLOTS DETECTED", "Upload a clearer, elevated view of a parking lot."
    elif r.available == 0:
        r.recommendation, r.advice = "PARKING FULL — TRY ANOTHER AREA", "Every detected slot is taken. Re-route to the nearest alternative lot."
    elif r.congestion == "HIGH":
        r.recommendation = "NEARLY FULL — TRY ANOTHER AREA"
        r.advice = (f"Only {r.available} slot(s) left. Unless you are already at the gate, "
                    f"search nearby lots. Last spaces are in the {r.best_zone} zone.")
    elif r.congestion == "MODERATE":
        r.recommendation = "SLOTS AVAILABLE — PROCEED"
        r.advice = f"{r.available} free slots. Head straight to the {r.best_zone} zone for the best chance."
    else:
        r.recommendation = "PLENTY OF SPACE — PROCEED"
        r.advice = f"{r.available} free slots. Lot is quiet, the {r.best_zone} zone has the most room."
    return r


def draw(bgr: np.ndarray, slots: list[Slot], show_ids=True) -> np.ndarray:
    out = bgr.copy()
    overlay = bgr.copy()
    th = max(1, round(min(bgr.shape[:2]) / 400))
    for s in slots:
        col = NEON_RED if s.occupied else NEON_GREEN
        x1, y1, x2, y2 = s.box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)
    out = cv2.addWeighted(overlay, 0.22, out, 0.78, 0)
    for s in slots:
        col = NEON_RED if s.occupied else NEON_GREEN
        x1, y1, x2, y2 = s.box
        cv2.rectangle(out, (x1, y1), (x2, y2), col, th + 1)
        if show_ids:
            fs = max(0.3, min(0.6, (x2 - x1) / 70))
            cv2.putText(out, str(s.id), (x1 + 2, y1 + int(14 * fs / 0.5)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.putText(out, str(s.id), (x1 + 2, y1 + int(14 * fs / 0.5)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return out
