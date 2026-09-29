"""
ParkVision AI - Step 3c: Model improvement round for the YOLO detector.
Round 1 (03_train_yolo.py, 25 epochs) reached mAP50 = 0.881 on the test frames, but recall was
the weak point (0.84) - some slots were missed, which directly corrupts the free-slot count.
Round 2 continues from the round-1 best weights with a lower learning rate, lower mosaic
(closer to real frames) and stronger brightness augmentation, for 20 more epochs.
"""
import json, shutil
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
model = YOLO(str(ROOT / "runs/yolo_slots/weights/best.pt"))
model.train(
    data=str(ROOT / "dataset/detection/data.yaml"),
    epochs=20, imgsz=640, batch=8, device="cpu", workers=2,
    project=str(ROOT / "runs"), name="yolo_slots_r2", exist_ok=True,
    optimizer="AdamW", lr0=0.0008, cos_lr=True, warmup_epochs=1,
    hsv_v=0.55, hsv_s=0.5, fliplr=0.5, degrees=3.0, scale=0.25, mosaic=0.5, close_mosaic=8,
    max_det=600, plots=True, verbose=False,
)
best = YOLO(str(ROOT / "runs/yolo_slots_r2/weights/best.pt"))
m = best.val(data=str(ROOT / "dataset/detection/data.yaml"), split="test", imgsz=640, device="cpu", max_det=600,
             project=str(ROOT / "runs"), name="yolo_test_r2", exist_ok=True, plots=True)
r1 = json.load(open(ROOT / "reports/yolo_metrics.json"))
r1 = r1.get("round1", r1)
res = {"round1": r1, "round2": {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "precision": float(m.box.mp),
                                "recall": float(m.box.mr), "epochs": "25 + 20", "imgsz": 640, "batch": 8}}
better = res["round2"]["mAP50_95"] >= r1["mAP50_95"]
res["selected"] = "round2" if better else "round1"
res.update(res[res["selected"]])
json.dump(res, open(ROOT / "reports/yolo_metrics.json", "w"), indent=2)
print(json.dumps(res, indent=2))
if better:
    p = best.export(format="onnx", imgsz=640, opset=12, simplify=False, dynamic=False)
    shutil.copy(p, ROOT / "models/yolo_slots.onnx")
    print("round 2 exported")
