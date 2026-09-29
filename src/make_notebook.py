"""Builds and executes ParkVision_AI_Notebook.ipynb (end-to-end walkthrough of the project)."""
import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# ParkVision AI — Intelligent Urban Parking Analytics & Space Optimisation Platform\n"
       "**Dhwanan Bhatt · 2505556 · Udgam School for Children · CRS Artificial Intelligence — Machine Learning & Deep Learning (SA)**\n\n"
       "This notebook walks through the full pipeline: problem definition → data preparation → model training & evaluation → "
       "insight logic → testing. Heavy training steps are in `src/*.py`; set `RUN_TRAINING = True` to re-run them "
       "(CPU training takes ~20 min for MobileNetV2 and ~45 min for YOLOv8n)."),
    md("## Step 1 · Problem & requirements\n"
       "| | |\n|---|---|\n| **Input** | Parking-lot image (elevated CCTV frame) |\n| **Output** | Slot-wise status (Occupied / Empty) + total / occupied / available |\n"
       "| **Layout** | PKLot lots are fixed, but rows are irregular (angled bays, trees, partial views) → a detector is used so no manual slot map is needed |\n"
       "| **Approach** | Hybrid: YOLOv8n full-image detection **+** MobileNetV2 crop classification |\n"
       "| **Challenges** | Shadows & lighting, occlusion by neighbouring cars/trees, sunny / cloudy / rainy weather |"),
    code("RUN_TRAINING = False\nimport json, subprocess, sys\nfrom pathlib import Path\nimport cv2, numpy as np, matplotlib.pyplot as plt\n"
         "ROOT = Path.cwd() if (Path.cwd()/'app.py').exists() else Path.cwd().parent\nprint(ROOT)"),
    md("## Step 2 · Data collection & preprocessing\nSource: PKLot (Kaggle `ammarnassanalhajali/pklot-dataset`). 121 full frames were sampled across 75 different days "
       "and five occupancy bands so that empty, half-full and full lots and all weather types are represented. "
       "`src/01_prepare_data.py` then:\n1. splits **frames** 70/15/15 (so crops from one frame never leak across splits),\n"
       "2. cuts every labelled slot out, resizes it to **224×224** and keeps a balanced **600 crops per class**,\n"
       "3. writes the YOLO-format detection set (`dataset/detection`)."),
    code("if RUN_TRAINING:\n    subprocess.run([sys.executable, str(ROOT/'src/01_prepare_data.py')], check=True)\n"
         "summary = json.load(open(ROOT/'reports/data_summary.json'))\nprint(json.dumps(summary, indent=1))\n"
         "for s in ['train','val','test']:\n    for c in ['empty','occupied']:\n        n = len(list((ROOT/'dataset/classification'/s/c).glob('*.jpg')))\n        print(f'{s:5s} {c:9s} {n}')"),
    code("fig, ax = plt.subplots(2, 1, figsize=(12, 5))\nfor a, f, t in zip(ax, ['sample_crops.png', 'augmentation_examples.png'],\n"
         "                  ['Slot crops (top: empty, bottom: occupied)', 'Augmentation: original, h-flip, rotate +15, rotate -15, brighter, darker']):\n"
         "    a.imshow(cv2.cvtColor(cv2.imread(str(ROOT/'reports'/f)), cv2.COLOR_BGR2RGB)); a.set_title(t); a.axis('off')\nplt.tight_layout(); plt.show()"),
    md("## Step 3a · MobileNetV2 slot classifier\nImageNet-pretrained MobileNetV2, first 10 blocks frozen, new 2-class head. 15 epochs, batch 32, AdamW (lr 1e-3) + cosine schedule, "
       "label smoothing 0.05. Train-time augmentation: random rotation ±15°, horizontal/vertical flip, brightness/contrast/saturation jitter, random resized crop."),
    code("if RUN_TRAINING:\n    subprocess.run([sys.executable, str(ROOT/'src/02_train_classifier.py'), '15'], check=True)\n"
         "m = json.load(open(ROOT/'reports/classifier_metrics.json'))\nprint('best val acc :', m['best_val_acc'])\nprint('TEST accuracy:', round(m['test_acc'], 4))\n"
         "print('confusion matrix [[TN FP] [FN TP]] (positive = occupied):', m['confusion_matrix'])\n"
         "for c in ['empty', 'occupied']:\n    r = m['report'][c]; print(f\"{c:9s} precision {r['precision']:.3f}  recall {r['recall']:.3f}  f1 {r['f1-score']:.3f}\")"),
    code("fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))\nfor a, f in zip(ax, ['training_curves.png', 'confusion_matrix.png']):\n"
         "    a.imshow(cv2.cvtColor(cv2.imread(str(ROOT/'reports'/f)), cv2.COLOR_BGR2RGB)); a.axis('off')\nplt.tight_layout(); plt.show()"),
    md("## Step 3b · YOLOv8n slot detector (+ improvement round)\nCOCO-pretrained YOLOv8n fine-tuned on full 640×640 frames with 2 classes (empty / occupied). "
       "Round 1: 25 epochs, batch 8, AdamW lr 0.002. Round 2 (improvement): +20 epochs from the best round-1 weights with lower lr, less mosaic and stronger brightness augmentation, "
       "targeting the weak recall of round 1."),
    code("if RUN_TRAINING:\n    subprocess.run([sys.executable, str(ROOT/'src/03_train_yolo.py'), '25'], check=True)\n"
         "    subprocess.run([sys.executable, str(ROOT/'src/03b_finetune_yolo.py')], check=True)\n"
         "y = json.load(open(ROOT/'reports/yolo_metrics.json'))\nfor k in ['round1', 'round2']:\n    if k in y: print(k, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in y[k].items() if kk != 'per_class_mAP50_95'})\n"
         "print('selected for the app ->', y.get('selected', 'round1'))"),
    md("## Step 4 · Integration & parking insight logic\nImplemented in `parkvision/engine.py`: YOLO finds slots → MobileNet re-checks each crop → probabilities fused → counts, occupancy %, "
       "congestion level (**Low < 40 %, Moderate 40–75 %, High > 75 %**), left/centre/right zone availability and a recommendation."),
    code("sys.path.insert(0, str(ROOT))\nfrom parkvision.engine import ParkVision\neng = ParkVision()\n"
         "fig, ax = plt.subplots(1, 3, figsize=(18, 6))\nfor a, name in zip(ax, ['02_morning_light_traffic', '05_overcast_half_full', '07_rush_hour_nearly_full']):\n"
         "    r = eng.scan(cv2.imread(str(ROOT/'samples'/f'{name}.jpg')), show_ids=False)\n"
         "    a.imshow(cv2.cvtColor(r.annotated, cv2.COLOR_BGR2RGB)); a.axis('off')\n"
         "    a.set_title(f'{name}\\n{r.total} slots | {r.occupied} occ | {r.available} free | {r.occupancy:.0f}% {r.congestion}\\n{r.recommendation}', fontsize=10)\n"
         "plt.tight_layout(); plt.show()"),
    md("## Step 6 · Testing on unseen frames & robustness\n`src/04_evaluate_pipeline.py` runs the complete app pipeline on the 18 held-out test frames and on artificially degraded versions "
       "(dark, glare, haze/rain-like low contrast, motion blur)."),
    code("if RUN_TRAINING:\n    subprocess.run([sys.executable, str(ROOT/'src/04_evaluate_pipeline.py')], check=True)\n"
         "ev = json.load(open(ROOT/'reports/pipeline_eval.json'))\nfor k in ['detector', 'classifier', 'hybrid']:\n"
         "    v = ev[k]; print(f\"{k:10s} slot acc {v['accuracy']:.4f} | recall {v['recall']:.3f} | MAE free-count {v['mae_available']:.2f}\")\n"
         "print('\\nrobustness (slot accuracy):')\nfor k, v in ev['robustness'].items(): print(f'  {k:28s}', v)"),
]
nbf.write(nb, ROOT / "ParkVision_AI_Notebook.ipynb")
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(nb, ROOT / "ParkVision_AI_Notebook.ipynb")
print("notebook written & executed")
