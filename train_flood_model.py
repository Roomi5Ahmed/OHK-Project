"""Train U-Net flood segmentation model on Sen1Floods11.

Usage:
    python train_flood_model.py --quick      # Quick sanity check: 5 batches, 1 epoch
    python train_flood_model.py --epochs 30  # Full training
    python train_flood_model.py              # Default: 30 epochs
"""
import sys
import csv
import argparse
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import segmentation_models_pytorch as smp
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_DIR = Path("C:/Git/OHK Project/data/sen1floods11")
MODEL_DIR = Path("C:/Git/OHK Project/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

S1_MEAN = [0.6851, 0.5235]
S1_STD = [0.0820, 0.1102]


class FloodDataset(Dataset):
    def __init__(self, csv_path, s1_dir, label_dir, augment=False, india_weight=3):
        self.s1_dir = s1_dir
        self.label_dir = label_dir
        self.augment = augment

        with open(csv_path) as f:
            reader = csv.reader(f)
            all_files = [(row[0], row[1]) for row in reader if len(row) >= 2]

        # Separate India and non-India chips
        self.india_files = [f for f in all_files if f[0].startswith('India')]
        self.other_files = [f for f in all_files if not f[0].startswith('India')]

        # Build weighted sample list: India chips repeated india_weight times
        self.files = self.other_files + self.india_files * india_weight
        self.has_india = len(self.india_files) > 0

        print(f"  Dataset: {len(all_files)} total, {len(self.india_files)} India (3x oversampled -> {len(self.files)} total)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        s1_name, label_name = self.files[idx]

        try:
            with rasterio.open(self.s1_dir / s1_name) as src:
                s1 = src.read()
            with rasterio.open(self.label_dir / label_name) as src:
                label = src.read(1)
        except Exception:
            return self.__getitem__(np.random.randint(len(self)))

        s1 = np.nan_to_num(s1, nan=-25.0)
        s1 = np.clip(s1, -50, 1)
        s1 = (s1 + 50) / 51.0
        s1 = (s1 - np.array(S1_MEAN)[:, None, None]) / np.array(S1_STD)[:, None, None]

        label = label.astype(np.float32)
        label[label == -1] = 255
        label = np.round(label)

        if self.augment:
            h, w = s1.shape[1], s1.shape[2]
            for _ in range(50):
                top = np.random.randint(0, h - 256 + 1)
                left = np.random.randint(0, w - 256 + 1)
                label_crop = label[top:top+256, left:left+256]
                valid = label_crop[label_crop != 255]
                if valid.size > 0 and len(np.unique(valid)) >= 2:
                    break
            s1 = s1[:, top:top+256, left:left+256]
            label = label_crop
            if np.random.random() > 0.5:
                s1 = np.flip(s1, axis=2).copy()
                label = np.flip(label, axis=1).copy()
            if np.random.random() > 0.5:
                s1 = np.flip(s1, axis=1).copy()
                label = np.flip(label, axis=0).copy()
        else:
            h, w = s1.shape[1], s1.shape[2]
            for _ in range(20):
                top = np.random.randint(0, h - 256 + 1)
                left = np.random.randint(0, w - 256 + 1)
                label_crop = label[top:top+256, left:left+256]
                if label_crop[label_crop != 255].size > 0:
                    break
            s1 = s1[:, top:top+256, left:left+256]
            label = label_crop

        return torch.tensor(s1, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def compute_iou(pred, target, ignore_index=255):
    pred = torch.argmax(pred, dim=1).flatten()
    target = target.flatten()
    mask = target.ne(ignore_index)
    pred = pred.masked_select(mask)
    target = target.masked_select(mask)
    if len(target) == 0:
        return 0.0
    intersection = (pred * target).sum().float()
    union = target.sum().float() + pred.sum().float() - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).item()


def compute_accuracy(pred, target, ignore_index=255):
    pred = torch.argmax(pred, dim=1).flatten()
    target = target.flatten()
    mask = target.ne(ignore_index)
    pred = pred.masked_select(mask)
    target = target.masked_select(mask)
    if len(target) == 0:
        return 0.0
    return (pred.eq(target).sum().float() / len(target)).item()


def train_one_epoch(model, loader, focal_loss, ce_loss, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0
    total_iou = 0
    total_acc = 0
    n = 0

    for i, (s1, label) in enumerate(tqdm(loader, desc="Training", leave=False)):
        if max_batches and i >= max_batches:
            break

        s1, label = s1.to(device), label.to(device)
        if label.ne(255).sum().item() == 0:
            continue

        optimizer.zero_grad()
        out = model(s1)
        loss = 0.5 * focal_loss(out, label) + 0.5 * ce_loss(out, label)

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        loss.backward()

        has_nan = False
        for p in model.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                has_nan = True
                break
        if has_nan:
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_iou += compute_iou(out, label)
        total_acc += compute_accuracy(out, label)
        n += 1

    return total_loss / max(n, 1), total_iou / max(n, 1), total_acc / max(n, 1)


def validate(model, loader, focal_loss, ce_loss, device, max_batches=None):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_acc = 0
    n = 0

    with torch.no_grad():
        for i, (s1, label) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            s1, label = s1.to(device), label.to(device)
            if label.ne(255).sum().item() == 0:
                continue
            out = model(s1)
            loss = 0.5 * focal_loss(out, label) + 0.5 * ce_loss(out, label)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            total_loss += loss.item()
            total_iou += compute_iou(out, label)
            total_acc += compute_accuracy(out, label)
            n += 1

    return total_loss / max(n, 1), total_iou / max(n, 1), total_acc / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test: 5 batches, 1 epoch")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--india-weight", type=int, default=3, help="Oversampling weight for India chips (default: 3)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"India chip oversampling: {args.india_weight}x")
    train_ds = FloodDataset(DATA_DIR / "flood_train_data.csv",
                             DATA_DIR / "s1", DATA_DIR / "labels", augment=True,
                             india_weight=args.india_weight)
    val_ds = FloodDataset(DATA_DIR / "flood_valid_data.csv",
                           DATA_DIR / "s1", DATA_DIR / "labels", augment=False,
                           india_weight=1)  # No oversampling for validation

    max_train_batches = 5 if args.quick else None
    max_val_batches = 3 if args.quick else None
    epochs = 1 if args.quick else args.epochs

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"Mode: {'QUICK TEST' if args.quick else f'FULL TRAINING ({epochs} epochs)'}")

    model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=2, classes=2)
    model = model.to(device)

    focal_loss = smp.losses.FocalLoss(mode="multiclass", ignore_index=255, alpha=0.25, gamma=2.0)
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 8.0]).to(device), ignore_index=255)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_iou = 0
    history = []

    for epoch in range(1, epochs + 1):
        t_loss, t_iou, t_acc = train_one_epoch(model, train_loader, focal_loss, ce_loss, optimizer, device, max_train_batches)
        scheduler.step()
        v_loss, v_iou, v_acc = validate(model, val_loader, focal_loss, ce_loss, device, max_val_batches)

        print(f"Epoch {epoch}/{epochs}: train_loss={t_loss:.4f} train_iou={t_iou:.4f} train_acc={t_acc:.4f} | val_loss={v_loss:.4f} val_iou={v_iou:.4f} val_acc={v_acc:.4f}")

        history.append({"epoch": epoch, "train_loss": t_loss, "train_iou": t_iou, "train_acc": t_acc,
                         "val_loss": v_loss, "val_iou": v_iou, "val_acc": v_acc})

        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(), MODEL_DIR / "flood_unet_best.pt")
            print(f"  -> Saved best model (val_iou={v_iou:.4f})")

    torch.save(model.state_dict(), MODEL_DIR / "flood_unet_final.pt")
    print(f"\nDone. Best val IoU: {best_iou:.4f}")

    import json
    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
