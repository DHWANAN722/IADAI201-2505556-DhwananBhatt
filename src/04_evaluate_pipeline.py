"""
ParkVision AI - Step 6: Testing on unseen frames
Runs the full app pipeline on the held-out test frames (never seen in training) and
compares three modes: YOLO only, MobileNet only (on YOLO boxes), and Hybrid fusion.
Also stress-tests robustness under artificial lighting changes (dark / bright / low-contrast / blur).
"""
import json, sys
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parkvision.engine import ParkVision

DET = ROOT / "dataset/detection"
eng = ParkVision()


def load_gt(stem, W, H):
    gt = []
    for line in (DET / "labels/test" / f"{stem}.txt").read_text().splitlines():
        c, cx, cy, w, h = map(float, line.split())
        gt.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H, int(c)))
    return gt


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def evaluate(transform=lambda x: x, modes=("detector", "classifier", "hybrid")):
    res = {m: {"correct": 0, "matched": 0, "gt": 0, "abs_err": []} for m in modes}
    for p in sorted((DET / "images/test").glob("*.jpg")):
        img = transform(cv2.imread(str(p)))
        H, W = img.shape[:2]
        gt = load_gt(p.stem, W, H)
        for m in modes:
            r = eng.scan(img, conf=0.25, mode=m)
            used = set()
            for s in r.slots:
                best, bj = 0, -1
                for j, g in enumerate(gt):
                    if j in used:
                        continue
                    v = iou(s.box, g)
                    if v > best:
                        best, bj = v, j
                if best >= 0.5:
                    used.add(bj); res[m]["matched"] += 1
                    res[m]["correct"] += int(s.occupied == bool(gt[bj][4]))
            res[m]["gt"] += len(gt)
            res[m]["abs_err"].append(abs(r.available - sum(1 for g in gt if g[4] == 0)))
    out = {}
    for m, v in res.items():
        out[m] = {"accuracy": v["correct"] / max(1, v["matched"]), "matched": v["matched"], "gt_slots": v["gt"],
                  "recall": v["matched"] / max(1, v["gt"]), "mae_available": float(np.mean(v["abs_err"]))}
    return out


report = evaluate()
print("clean:", json.dumps(report, indent=1))
stress = {
    "dark (-45% brightness)": lambda x: cv2.convertScaleAbs(x, alpha=0.55, beta=0),
    "bright / glare (+60)": lambda x: cv2.convertScaleAbs(x, alpha=1.0, beta=60),
    "low contrast (haze/rain)": lambda x: cv2.addWeighted(x, 0.5, np.full_like(x, 128), 0.5, 0),
    "motion blur": lambda x: cv2.filter2D(x, -1, np.eye(9) / 9),
}
report["robustness"] = {}
for name, fn in stress.items():
    r = evaluate(fn, modes=("detector", "hybrid"))
    report["robustness"][name] = {m: round(v["accuracy"], 4) for m, v in r.items()} | {"recall": round(r["hybrid"]["recall"], 4)}
    print(name, report["robustness"][name])
json.dump(report, open(ROOT / "reports/pipeline_eval.json", "w"), indent=2)
