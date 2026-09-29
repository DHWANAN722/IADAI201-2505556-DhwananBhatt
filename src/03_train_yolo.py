"""
ParkVision AI - Step 3b: YOLOv8n slot detector (full-image detection)
Detects every parking slot in a full frame and labels it empty / occupied.
Transfer learning from COCO-pretrained yolov8n.pt, then exported to ONNX for the web app.
"""
import json, shutil, sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 25

model = YOLO(str(ROOT / "yolov8n.pt"))
model.train(
    data=str(ROOT / "dataset/detection/data.yaml"),
    epochs=EPOCHS, imgsz=640, batch=8, device="cpu", workers=2,
    project=str(ROOT / "runs"), name="yolo_slots", exist_ok=True,
    optimizer="AdamW", lr0=0.002, cos_lr=True, patience=50,
    # augmentation: brightness/HSV, flips, small rotations & scale (robust to weather/lighting)
    hsv_v=0.45, hsv_s=0.5, fliplr=0.5, degrees=5.0, scale=0.3, mosaic=1.0, close_mosaic=5,
    max_det=600, plots=True, verbose=False,
)
best = YOLO(str(ROOT / "runs/yolo_slots/weights/best.pt"))
metrics = best.val(data=str(ROOT / "dataset/detection/data.yaml"), split="test", imgsz=640,
                   device="cpu", max_det=600, project=str(ROOT / "runs"), name="yolo_test", exist_ok=True, plots=True)
res = {
    "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
    "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
    "per_class_mAP50_95": {n: float(v) for n, v in zip(["empty", "occupied"], metrics.box.maps)},
    "epochs": EPOCHS, "imgsz": 640, "batch": 8,
}
(ROOT / "reports").mkdir(exist_ok=True)
json.dump(res, open(ROOT / "reports/yolo_metrics.json", "w"), indent=2)
print(res)
onnx_path = best.export(format="onnx", imgsz=640, opset=12, simplify=False, dynamic=False)
(ROOT / "models").mkdir(exist_ok=True)
shutil.copy(onnx_path, ROOT / "models/yolo_slots.onnx")
print("exported", onnx_path)
