"""
ParkVision AI - Step 3a: MobileNetV2 slot classifier (crop-based classification)
Input : 224x224 slot crops  (dataset/classification/{train,val,test}/{empty,occupied})
Output: models/slot_classifier.onnx, reports/classifier_metrics.json,
        reports/training_curves.png, reports/confusion_matrix.png

Transfer learning: ImageNet-pretrained MobileNetV2, new 2-class head.
Augmentation (train only): random rotation +-15 deg, horizontal/vertical flip,
brightness/contrast jitter (simulates sun, shadow, rain), small random resized crop.
"""
import json, time, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42); np.random.seed(42)
torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "classification"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
BATCH = 32
LR = 1e-3
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

ds = {s: datasets.ImageFolder(DATA / s, train_tf if s == "train" else eval_tf) for s in ("train", "val", "test")}
dl = {s: DataLoader(d, batch_size=BATCH, shuffle=(s == "train"), num_workers=2) for s, d in ds.items()}
classes = ds["train"].classes  # ['empty', 'occupied']
print(classes, {s: len(d) for s, d in ds.items()})

model = models.mobilenet_v2()
model.load_state_dict(torch.load(ROOT / "mobilenet_v2-b0353104.pth", map_location="cpu"))
# freeze the first 10 feature blocks (generic edges/textures), fine-tune the rest
for p in model.features[:10].parameters():
    p.requires_grad = False
model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.last_channel, 2))

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit = nn.CrossEntropyLoss(label_smoothing=0.05)

def run(split, train=False):
    model.train(train)
    tot, correct, loss_sum, ys, ps = 0, 0, 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in dl[split]:
            out = model(x); loss = crit(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            pred = out.argmax(1)
            tot += len(y); correct += (pred == y).sum().item(); loss_sum += loss.item() * len(y)
            ys += y.tolist(); ps += pred.tolist()
    return loss_sum / tot, correct / tot, ys, ps

hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_acc, t0 = 0, time.time()
for ep in range(1, EPOCHS + 1):
    tl, ta, _, _ = run("train", True)
    vl, va, _, _ = run("val")
    sched.step()
    for k, v in zip(hist, (tl, ta, vl, va)):
        hist[k].append(v)
    if va >= best_acc:
        best_acc = va; torch.save(model.state_dict(), ROOT / "models" / "slot_classifier.pt")
    print(f"epoch {ep:02d}/{EPOCHS}  train_loss {tl:.4f} acc {ta:.4f} | val_loss {vl:.4f} acc {va:.4f}  [{time.time()-t0:.0f}s]", flush=True)

model.load_state_dict(torch.load(ROOT / "models" / "slot_classifier.pt"))
_, test_acc, ys, ps = run("test")
cm = confusion_matrix(ys, ps)
rep = classification_report(ys, ps, target_names=classes, output_dict=True)
print("TEST ACC", test_acc); print(cm)

# ---- plots ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
e = range(1, EPOCHS + 1)
ax[0].plot(e, hist["train_acc"], label="train"); ax[0].plot(e, hist["val_acc"], label="val"); ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(e, hist["train_loss"], label="train"); ax[1].plot(e, hist["val_loss"], label="val"); ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig(ROOT / "reports/training_curves.png", dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1], classes); ax.set_yticks([0, 1], classes)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Test confusion matrix (acc {test_acc:.3f})")
plt.tight_layout(); plt.savefig(ROOT / "reports/confusion_matrix.png", dpi=120); plt.close()

json.dump({"classes": classes, "epochs": EPOCHS, "batch_size": BATCH, "lr": LR, "optimizer": "AdamW + cosine",
           "best_val_acc": best_acc, "test_acc": test_acc, "confusion_matrix": cm.tolist(),
           "report": rep, "history": hist}, open(ROOT / "reports/classifier_metrics.json", "w"), indent=2)

# ---- ONNX export for the Streamlit app ----
model.eval()
torch.onnx.export(model, torch.randn(1, 3, 224, 224), ROOT / "models/slot_classifier.onnx",
                  input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=13, dynamo=False)
print("saved models/slot_classifier.onnx")
