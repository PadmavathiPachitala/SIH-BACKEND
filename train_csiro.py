import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm

# =====================================================================
# CONFIGURATION & HYPERPARAMETERS (EASY TO MODIFY AT THE TOP)
# =====================================================================
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
IMAGE_SIZE = 224  # 224 is standard EfficientNet input; 400 is too slow on CPU
SEED = 42
NUM_WORKERS = 0  # Required for Windows compatibility

OUTPUT_DIR = "outputs"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "csiro_best.pt")
METRICS_PATH = os.path.join(OUTPUT_DIR, "csiro_test_metrics.txt")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_csiro_dataset_root(base_dir="."):
    """
    Automatically detects the CSIRO dataset directory containing class 0 (no oil) and class 1 (oil).
    Checks standard folder structures first, then searches the workspace if needed.
    """
    candidate_paths = [
        os.path.join(base_dir, "data", "csiro"),
        os.path.join(base_dir, "DATASETS", "kaggle", "data"),
        os.path.join(base_dir, "DATASETS", "csiro"),
        os.path.join(base_dir, "data"),
    ]

    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        subdirs_lower = [d.lower() for d in subdirs]

        if ("0" in subdirs and "1" in subdirs) or \
           ("class_0" in subdirs_lower and "class_1" in subdirs_lower) or \
           ("no_oil" in subdirs_lower and "oil" in subdirs_lower) or \
           ("no-oil" in subdirs_lower and "oil" in subdirs_lower):
            return os.path.abspath(path)

    # Recursive fallback search across the workspace
    for root, dirs, _ in os.walk(base_dir):
        if any(skip in root.lower() for skip in [".git", "venv", "outputs", ".gemini", "node_modules"]):
            continue
        dirs_lower = [d.lower() for d in dirs]
        if ("class_0" in dirs_lower and "class_1" in dirs_lower) or \
           ("0" in dirs and "1" in dirs and "csiro" in root.lower()):
            return os.path.abspath(root)

    raise FileNotFoundError(
        "CSIRO dataset root not found. Please ensure images are placed in data/csiro/0/ and data/csiro/1/ "
        "or DATASETS/kaggle/data/Class_0 and DATASETS/kaggle/data/Class_1."
    )


def get_image_paths_and_labels(dataset_root):
    """
    Scans dataset root for class folders and maps images to labels:
    Label 0: No Oil (folders: 0, class_0, no_oil, no-oil)
    Label 1: Oil    (folders: 1, class_1, oil)
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    image_paths = []
    labels = []

    subdirs = os.listdir(dataset_root)
    class_0_dir = None
    class_1_dir = None

    for d in subdirs:
        full_p = os.path.join(dataset_root, d)
        if not os.path.isdir(full_p):
            continue
        d_lower = d.lower()
        if d == "0" or d_lower in ("class_0", "no_oil", "no-oil", "no_oil_spill"):
            class_0_dir = full_p
        elif d == "1" or d_lower in ("class_1", "oil", "oil_spill"):
            class_1_dir = full_p

    if not class_0_dir or not class_1_dir:
        raise ValueError(f"Could not identify Class 0 (no-oil) and Class 1 (oil) directories inside: {dataset_root}")

    for img_name in sorted(os.listdir(class_0_dir)):
        ext = os.path.splitext(img_name)[1].lower()
        if ext in valid_exts:
            image_paths.append(os.path.join(class_0_dir, img_name))
            labels.append(0)

    for img_name in sorted(os.listdir(class_1_dir)):
        ext = os.path.splitext(img_name)[1].lower()
        if ext in valid_exts:
            image_paths.append(os.path.join(class_1_dir, img_name))
            labels.append(1)

    print("==================================================")
    print(f"CSIRO Dataset Root Found: {dataset_root}")
    print(f"Class 0 (No Oil): {labels.count(0)} images")
    print(f"Class 1 (Oil):    {labels.count(1)} images")
    print(f"Total Images:     {len(labels)}")
    print("==================================================")

    return image_paths, labels


class CSIRODataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        # 1-channel grayscale image
        image = Image.open(path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(img_size=IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    return train_transform, val_test_transform


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in tqdm(dataloader, desc="  train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, pos_label=1, average='binary', zero_division=0)
    return epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in tqdm(dataloader, desc="  eval ", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, pos_label=1, average='binary', zero_division=0)
    return acc, f1, np.array(all_targets), np.array(all_preds)


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Dataset Detection & Image Listing
    dataset_root = find_csiro_dataset_root()
    image_paths, labels = get_image_paths_and_labels(dataset_root)

    # 2. Stratified Split: 80% train, 10% val, 10% test
    # Step 1: Split 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.20, random_state=SEED, stratify=labels
    )
    # Step 2: Split 20% temp into 50% val (10% total) and 50% test (10% total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    print(f"Dataset Split Summary:")
    print(f"  Train Set: {len(X_train)} samples")
    print(f"  Val Set:   {len(X_val)} samples")
    print(f"  Test Set:  {len(X_test)} samples")

    # 3. Data Transforms & Loaders
    train_tf, val_tf = get_transforms(IMAGE_SIZE)

    train_dataset = CSIRODataset(X_train, y_train, transform=train_tf)
    val_dataset = CSIRODataset(X_val, y_val, transform=val_tf)
    test_dataset = CSIRODataset(X_test, y_test, transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 4. Model Setup (timm efficientnet_b0 with 1-channel input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing Device: {device}")

    # mobilenetv3_small_100: ~2.5M params, ~10x faster than efficientnet_b0 on CPU
    model = timm.create_model("mobilenetv3_small_100", pretrained=True, in_chans=1, num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop with Val F1 Checkpointing
    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_epoch = 0

    print(f"\nStarting Training for {EPOCHS} Epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] | "
            f"Train Acc: {train_acc:.4f} | Train F1 (Oil): {train_f1:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1 (Oil): {val_f1:.4f}"
        )

        # Save ONLY when validation Oil F1 improves
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch

            checkpoint = {
                "state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "epoch": epoch,
            }
            torch.save(checkpoint, BEST_MODEL_PATH)
            print(f"  --> Saved BEST checkpoint to {BEST_MODEL_PATH} (Val F1: {val_f1:.4f})")

    print(f"\nTraining Complete. Best Validation Oil F1: {best_val_f1:.4f} at Epoch {best_epoch}.")

    # 6. Evaluate Best Checkpoint on TEST SET ONCE
    print("\nLoading Best Model Checkpoint for Single Test Set Evaluation...")
    best_checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"])

    test_acc, test_oil_f1, y_test_true, y_test_pred = evaluate(model, test_loader, criterion, device)
    report = classification_report(y_test_true, y_test_pred, target_names=["No Oil (0)", "Oil (1)"], digits=4)
    cm = confusion_matrix(y_test_true, y_test_pred)

    print("\n" + "=" * 50)
    print("      TEST SET CLASSIFICATION REPORT")
    print("=" * 50)
    print(report)
    print("=" * 50)
    print("         TEST SET CONFUSION MATRIX")
    print("=" * 50)
    print(cm)
    print("=" * 50)

    # 7. Write Test Metrics Summary File
    with open(METRICS_PATH, "w") as f:
        f.write(f"test_acc: {test_acc:.6f}\n")
        f.write(f"test_oil_f1: {test_oil_f1:.6f}\n")
        f.write(f"val_best_f1: {best_checkpoint['val_f1']:.6f}\n")
        f.write(f"best_epoch: {best_checkpoint['epoch']}\n")

    print(f"\nSaved test metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
