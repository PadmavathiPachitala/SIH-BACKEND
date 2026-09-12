# SIH26143 Oil Spill Detection — Two-Stage Pipeline

## Stage 1: CSIRO Binary Classifier (`train_csiro.py`)
Image-level classification: oil vs no-oil on CSIRO Sentinel-1 SAR patches.
- Primary metric: **Oil-class F1** (not pixel accuracy)
- Output: `outputs/csiro_best.pt`

## Stage 2: U-Net Segmentation (`train_unet.py`)
Pixel-level segmentation: SAR image → binary oil mask.
- Primary metrics: **Dice coefficient** and **IoU** (not accuracy)
- Output: `outputs/unet_best.pt`, `outputs/unet_test_metrics.txt`, `outputs/unet_previews/`

---

# CSIRO Sentinel-1 SAR Oil / No-Oil Binary Classifier

## Overview & Background

The **CSIRO Sentinel-1 SAR Oil/No-Oil Dataset** consists of grayscale Synthetic Aperture Radar (SAR) patch images (`~400x400` resolution) captured by Sentinel-1 satellites. 

This project trains a **binary image classifier** (`oil` vs `no-oil`), distinguishing SAR sea surface slick signatures from clean sea clutter/look-alikes.

> [!IMPORTANT]
> **Key Operational Constraints**:
> - **Binary Classification Only**: Task is standard image-level binary classification (oil vs. no-oil), **not** U-Net semantic pixel segmentation.
> - **Dataset Isolation**: CSIRO dataset contains image-level labels without pixel masks. Do not attempt U-Net segmentation on CSIRO, and do not mix CSIRO images with Arabian Sea GeoTIFF segmentation rasters.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset Structure
The dataset auto-detector automatically discovers the data path in your repository. Standard directory layout:
```text
data/csiro/
├── 0/    # Class 0: No Oil
└── 1/    # Class 1: Oil
```
*(Alternative directory names such as `DATASETS/kaggle/data/Class_0` and `DATASETS/kaggle/data/Class_1` or `oil`/`no_oil` are also automatically recognized).*

### 3. Start Training
```bash
python train_csiro.py
```

- **Default Hyperparameters** (configurable at top of `train_csiro.py`):
  - `EPOCHS = 15`
  - `BATCH_SIZE = 32` (if CUDA OOM occurs, change to `16` or `8`)
  - `LEARNING_RATE = 3e-4`
  - `IMAGE_SIZE = 400`
  - `SEED = 42`

---

## Model Architecture & Pipeline Highlights

- **Model**: `timm.create_model('efficientnet_b0', pretrained=True, in_chans=1, num_classes=2)`
  - Single-channel grayscale input tensor `(1, 400, 400)`
- **Data Augmentations**:
  - Training: `Resize(400, 400)`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(90)`, `Normalize(mean=0.5, std=0.5)`
  - Validation/Test: `Resize(400, 400)`, `Normalize(mean=0.5, std=0.5)`
- **Splits**: Stratified 80% Train, 10% Validation, 10% Test (`seed=42`)
- **Checkpointing**: Saves **ONLY** when Validation Oil-Class F1 score improves (`outputs/csiro_best.pt`). Prevents overfitting on late epochs.
- **Single Test Evaluation**: Evaluates test set exactly once using `outputs/csiro_best.pt` after training ends.

---

## Output Artifacts

- **Model Checkpoint**: `outputs/csiro_best.pt`
  - Contains: `{state_dict, val_f1, val_acc, epoch}`
- **Test Set Summary**: `outputs/csiro_test_metrics.txt`
  - Contains: `test_acc`, `test_oil_f1`, `val_best_f1`, `best_epoch`

---

## Metric Guide for Judges & Evaluators

In real-world marine environmental monitoring and emergency response, **raw pixel accuracy is misleading**:
1. **Class Imbalance**: Clean sea surfaces vastly outnumber oil slicks in SAR imagery. A dummy model predicting "no-oil" for all images achieves high accuracy while missing 100% of oil spills.
2. **Oil-Class F1 Score (Primary Metric)**:
   - **Precision**: Minimizes false alarms (sending emergency response vessels for natural wind low-lookalikes).
   - **Recall**: Minimizes missed spill detections (catastrophic unaddressed oil contamination).
   - **F1 Score** measures the harmonic mean of Precision and Recall specifically on the target **Oil Class (`pos_label=1`)**, providing judges with a rigorous, un-biased assessment of detection capability.
