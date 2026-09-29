"""
ParkVision AI - Step 2: Data collection & preprocessing
-------------------------------------------------------
Source : PKLot dataset (Kaggle: ammarnassanalhajali/pklot-dataset, Roboflow COCO export).
Input  : data/raw/*.jpg  (121 full parking-lot frames, 640x640, sunny/cloudy/rainy)
         data/_annotations.coco.json  (bounding boxes: space-empty / space-occupied)

Outputs
  1. dataset/classification/{train,val,test}/{empty,occupied}/*.jpg   (slot crops, 224x224)
  2. dataset/detection/{images,labels}/{train,val,test}  + data.yaml     (YOLO format)
  3. reports/data_summary.json, reports/sample_crops.png, reports/augmentation_examples.png

The split is done at the *frame* level (70/15/15) so crops from the same frame never
leak between train / val / test.
"""
import json, random, shutil, collections
from pathlib import Path
import cv2
import numpy as np

random.seed(42)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ANN = ROOT / "data" / "_annotations.coco.json"
OUT_CLS = ROOT / "dataset" / "classification"
OUT_DET = ROOT / "dataset" / "detection"
REPORTS = ROOT / "reports"
IMG_SIZE = 224
CROPS_PER_CLASS = {"train": 420, "val": 90, "test": 90}   # 600 per class in total (>=100 required)
CLASS_NAMES = {1: "empty", 2: "occupied"}

for p in (OUT_CLS, OUT_DET):
    shutil.rmtree(p, ignore_errors=True)
REPORTS.mkdir(parents=True, exist_ok=True)

coco = json.load(open(ANN))
available = {p.name for p in RAW.glob("*.jpg")}
images = {im["id"]: im for im in coco["images"] if im["file_name"] in available}
anns = collections.defaultdict(list)
for a in coco["annotations"]:
    if a["image_id"] in images and a["category_id"] in CLASS_NAMES:
        anns[a["image_id"]].append(a)

# ---------------- frame-level 70 / 15 / 15 split ----------------
ids = sorted(images)
random.shuffle(ids)
n = len(ids)
n_tr, n_va = round(n * 0.70), round(n * 0.15)
split_ids = {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}

summary = {"frames": {k: len(v) for k, v in split_ids.items()}, "slots": {}, "crops": {}}

# ---------------- 1. classification crops ----------------
def crop_slot(img, bbox, pad=0.08):
    x, y, w, h = bbox
    H, W = img.shape[:2]
    px, py = w * pad, h * pad
    x0, y0 = int(max(0, x - px)), int(max(0, y - py))
    x1, y1 = int(min(W, x + w + px)), int(min(H, y + h + py))
    c = img[y0:y1, x0:x1]
    return cv2.resize(c, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA) if c.size else None

sample_tiles = {"empty": [], "occupied": []}
for split, sids in split_ids.items():
    pool = {"empty": [], "occupied": []}
    for iid in sids:
        for a in anns[iid]:
            if a["bbox"][2] < 8 or a["bbox"][3] < 8:
                continue  # drop degenerate boxes (label cleaning)
            pool[CLASS_NAMES[a["category_id"]]].append((iid, a["bbox"]))
    summary["slots"][split] = {k: len(v) for k, v in pool.items()}
    summary["crops"][split] = {}
    cache = {}
    for cls, items in pool.items():
        random.shuffle(items)
        chosen = items[:CROPS_PER_CLASS[split]]
        d = OUT_CLS / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for k, (iid, bbox) in enumerate(chosen):
            if iid not in cache:
                cache[iid] = cv2.imread(str(RAW / images[iid]["file_name"]))
            c = crop_slot(cache[iid], bbox)
            if c is None:
                continue
            cv2.imwrite(str(d / f"{cls}_{split}_{k:04d}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if split == "train" and len(sample_tiles[cls]) < 8:
                sample_tiles[cls].append(c)
        summary["crops"][split][cls] = len(list(d.glob("*.jpg")))

# ---------------- 2. YOLO detection dataset ----------------
for split, sids in split_ids.items():
    (OUT_DET / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_DET / "labels" / split).mkdir(parents=True, exist_ok=True)
    for iid in sids:
        im = images[iid]
        W, H = im["width"], im["height"]
        shutil.copy(RAW / im["file_name"], OUT_DET / "images" / split / im["file_name"])
        lines = []
        for a in anns[iid]:
            x, y, w, h = a["bbox"]
            if w < 4 or h < 4:
                continue
            cls = 0 if a["category_id"] == 1 else 1
            lines.append(f"{cls} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        (OUT_DET / "labels" / split / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
(OUT_DET / "data.yaml").write_text(
    f"path: {OUT_DET}\ntrain: images/train\nval: images/val\ntest: images/test\n"
    "names:\n  0: empty\n  1: occupied\n")

# ---------------- 3. visual reports ----------------
def tile(row):
    return np.hstack([cv2.resize(r, (112, 112)) for r in row])
grid = np.vstack([tile(sample_tiles["empty"]), tile(sample_tiles["occupied"])])
cv2.imwrite(str(REPORTS / "sample_crops.png"), grid)

# augmentation preview (same ops used in training: rotation, flip, brightness)
base = sample_tiles["occupied"][0]
def rot(img, deg):
    M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT)
augs = [base, cv2.flip(base, 1), rot(base, 15), rot(base, -15),
        cv2.convertScaleAbs(base, alpha=1.0, beta=45), cv2.convertScaleAbs(base, alpha=0.6, beta=0)]
cv2.imwrite(str(REPORTS / "augmentation_examples.png"), tile(augs))

json.dump(summary, open(REPORTS / "data_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
