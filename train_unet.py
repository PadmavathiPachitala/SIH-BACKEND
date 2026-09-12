import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
EPOCHS      = 30
BATCH_SIZE  = 8       # drop to 4 if OOM
LR          = 1e-4
IMG_SIZE    = 128     # 256 is too slow on CPU; 128 gives ~8x speedup
SEED        = 42
NUM_WORKERS = 0       # Windows

IMG_ROOT  = os.path.join("data", "images", "images")
MASK_ROOT = os.path.join("data", "masks",  "masks")
OUT_DIR   = "outputs"
BEST_PT   = os.path.join(OUT_DIR, "unet_best.pt")
METRICS_F = os.path.join(OUT_DIR, "unet_test_metrics.txt")
PREV_DIR  = os.path.join(OUT_DIR, "unet_previews")
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def collect_pairs(img_root, mask_root):
    """Walk all sub-folders of img_root; match masks by stem."""
    img_map = {}
    for root, _, files in os.walk(img_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXT:
                img_map[os.path.splitext(f)[0]] = os.path.join(root, f)

    pairs = []
    for root, _, files in os.walk(mask_root):
        for f in files:
            stem = os.path.splitext(f)[0]
            if stem in img_map and os.path.splitext(f)[1].lower() in VALID_EXT:
                pairs.append((img_map[stem], os.path.join(root, f)))

    pairs.sort(key=lambda x: x[0])
    return pairs


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(path).convert("L"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img[None]  # (1, H, W)


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        mask = np.array(Image.open(path))
    # If multi-channel (color mask), collapse to single channel
    if mask.ndim == 3:
        mask = mask[..., 0]  # take first channel; oil pixels are non-zero
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    # Binarise: any non-zero value → oil (1)
    binary = (mask > 0).astype(np.float32)
    return binary[None]  # (1, H, W)


def augment(img, mask):
    """Identical spatial augmentation on image and mask."""
    if random.random() > 0.5:
        img = np.flip(img, axis=2).copy()
        mask = np.flip(mask, axis=2).copy()
    if random.random() > 0.5:
        img = np.flip(img, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
    k = random.randint(0, 3)
    if k:
        img = np.rot90(img, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(1, 2)).copy()
    return img, mask


class OilSpillDataset(Dataset):
    def __init__(self, pairs, train=False):
        self.pairs = pairs
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img  = load_image(self.pairs[idx][0])
        mask = load_mask(self.pairs[idx][1])
        if self.train:
            img, mask = augment(img, mask)
        return torch.from_numpy(img), torch.from_numpy(mask)


def dice_iou(pred_logits, target, thresh=0.5):
    pred = (torch.sigmoid(pred_logits) > thresh).float()
    p, t = pred.view(-1), target.view(-1)
    inter = (p * t).sum()
    dice = (2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
    iou  = (inter + 1e-6) / (p.sum() + t.sum() - inter + 1e-6)
    return dice.item(), iou.item()


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss = total_dice = total_iou = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, masks in tqdm(loader, desc="  train" if train else "  eval ", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            d, i = dice_iou(logits, masks)
            total_loss += loss.item(); total_dice += d; total_iou += i
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def save_previews(model, pairs, device, n=8):
    os.makedirs(PREV_DIR, exist_ok=True)
    model.eval()
    indices = random.sample(range(len(pairs)), min(n, len(pairs)))
    with torch.no_grad():
        for k, idx in enumerate(indices):
            img_raw  = cv2.imread(pairs[idx][0], cv2.IMREAD_GRAYSCALE)
            mask_raw = load_mask(pairs[idx][1])[0]  # (H,W)
            img_t = torch.from_numpy(load_image(pairs[idx][0])).unsqueeze(0).to(device)
            pred  = (torch.sigmoid(model(img_t)) > 0.5).float().cpu().numpy()[0, 0]

            img_disp  = cv2.resize(img_raw,  (IMG_SIZE, IMG_SIZE))
            img_rgb   = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)
            true_rgb  = cv2.cvtColor((mask_raw * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            pred_rgb  = cv2.cvtColor((pred     * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            row = np.concatenate([img_rgb, true_rgb, pred_rgb], axis=1)
            cv2.imwrite(os.path.join(PREV_DIR, f"preview_{k:02d}.png"), row)


def main():
    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)

    pairs = collect_pairs(IMG_ROOT, MASK_ROOT)
    print(f"Found {len(pairs)} image-mask pairs")
    assert len(pairs) > 0, f"No pairs found. Check IMG_ROOT={IMG_ROOT} and MASK_ROOT={MASK_ROOT}"

    stems = [os.path.splitext(os.path.basename(p[0]))[0] for p in pairs]
    tr_s, tmp_s = train_test_split(stems, test_size=0.20, random_state=SEED)
    va_s, te_s  = train_test_split(tmp_s,  test_size=0.50, random_state=SEED)
    stem_set = {s: p for s, p in zip(stems, pairs)}
    tr_pairs = [stem_set[s] for s in tr_s]
    va_pairs = [stem_set[s] for s in va_s]
    te_pairs = [stem_set[s] for s in te_s]
    print(f"Split → train:{len(tr_pairs)}  val:{len(va_pairs)}  test:{len(te_pairs)}")

    tr_loader = DataLoader(OilSpillDataset(tr_pairs, train=True),  batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    va_loader = DataLoader(OilSpillDataset(va_pairs, train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    te_loader = DataLoader(OilSpillDataset(te_pairs, train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = smp.Unet(
        encoder_name="mobilenet_v2",   # ~3.4M params vs resnet34's 21M; ~6x faster on CPU
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ).to(device)

    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss  = nn.BCEWithLogitsLoss()
    criterion = lambda logits, masks: dice_loss(logits, masks) + bce_loss(logits, masks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_dice = -1.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_dice, tr_iou = run_epoch(model, tr_loader, criterion, optimizer, device, train=True)
        va_loss, va_dice, va_iou = run_epoch(model, va_loader, criterion, optimizer, device, train=False)
        print(f"Epoch [{epoch:02d}/{EPOCHS}] "
              f"| Train Loss:{tr_loss:.4f} Dice:{tr_dice:.4f} IoU:{tr_iou:.4f} "
              f"| Val Loss:{va_loss:.4f} Dice:{va_dice:.4f} IoU:{va_iou:.4f}")
        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({"state_dict": model.state_dict(),
                        "val_dice": va_dice, "val_iou": va_iou, "epoch": epoch}, BEST_PT)
            print(f"  --> Saved best checkpoint (Val Dice: {va_dice:.4f})")

    print("\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(BEST_PT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    _, te_dice, te_iou = run_epoch(model, te_loader, criterion, optimizer, device, train=False)
    print(f"\nTEST  Dice: {te_dice:.4f}  IoU: {te_iou:.4f}")
    print(f"Best Val  Dice: {ckpt['val_dice']:.4f}  IoU: {ckpt['val_iou']:.4f}  Epoch: {ckpt['epoch']}")

    with open(METRICS_F, "w") as f:
        f.write(f"test_dice: {te_dice:.6f}\n")
        f.write(f"test_iou: {te_iou:.6f}\n")
        f.write(f"val_best_dice: {ckpt['val_dice']:.6f}\n")
        f.write(f"val_best_iou: {ckpt['val_iou']:.6f}\n")
        f.write(f"best_epoch: {ckpt['epoch']}\n")
    print(f"Saved metrics → {METRICS_F}")

    save_previews(model, te_pairs, device)
    print(f"Saved 8 preview PNGs → {PREV_DIR}/")


if __name__ == "__main__":
    main()
