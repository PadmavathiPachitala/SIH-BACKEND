"""
infer_geotiff.py — U-Net inference on real Arabian Sea Sentinel-1 GeoTIFFs.
Matches train_unet.py architecture and normalization:
- segmentation_models_pytorch Unet with mobilenet_v2 encoder
- in_channels=1, classes=1
- loads outputs/unet_best.pt
- reads GeoTIFFs with rasterio, preserves CRS and transform
- performs tiled inference with overlap and stitching
- saves georeferenced binary masks, overlay PNGs, preview PNGs, and summary.json
"""

import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import cv2
import torch
import rasterio
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Optimize CPU multithreading
torch.set_num_threads(max(1, os.cpu_count() or 4))

# ── CONFIG ───────────────────────────────────────────────────────────────────
ENCODER      = "mobilenet_v2"   # Matches train_unet.py
IN_CHANNELS  = 1
CLASSES      = 1
TILE_SIZE    = 256
OVERLAP      = 32
THRESHOLD    = 0.5
MIN_OIL_FRAC = 0.0001          # 0.01% of total scene pixels

CHECKPOINT   = os.path.join("outputs", "unet_best.pt")
INPUT_DIR    = os.path.join("data", "arabian_sea_geotiff")

OUT_ROOT     = "inference_outputs"
MASK_DIR     = os.path.join(OUT_ROOT, "masks")
OVERLAY_DIR  = os.path.join(OUT_ROOT, "overlays")
PREVIEW_DIR  = os.path.join(OUT_ROOT, "previews")
SUMMARY_F    = os.path.join(OUT_ROOT, "summary.json")
# ─────────────────────────────────────────────────────────────────────────────


def load_model(checkpoint_path, device):
    """Loads U-Net model matching train_unet.py specs and checkpoint weights."""
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,   # Inference only
        in_channels=IN_CHANNELS,
        classes=CLASSES,
    ).to(device)

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # Strip 'module.' prefix if saved from DataParallel
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_crop(band):
    """
    Finds bounding box of valid SAR data, crops array to minimize memory footprint,
    and normalizes floating point SAR band (dB) to uint8 preview and [-1, 1] model input.
    Matches load_image in train_unet.py: (val / 255.0 - 0.5) / 0.5.
    """
    H, W = band.shape
    valid_mask_full = (band != 0) & (~np.isnan(band)) & np.isfinite(band)

    if not np.any(valid_mask_full):
        return None, None, False, 0, 0, 0, 0, H, W

    rows, cols = np.where(valid_mask_full)
    ymin, ymax = int(rows.min()), int(rows.max())
    xmin, xmax = int(cols.min()), int(cols.max())

    crop_band = band[ymin:ymax+1, xmin:xmax+1]
    crop_valid = valid_mask_full[ymin:ymax+1, xmin:xmax+1]

    valid_vals = crop_band[crop_valid]
    p2, p98 = np.percentile(valid_vals, 2), np.percentile(valid_vals, 98)
    denom = max(float(p98 - p2), 1e-6)

    crop_u8 = np.zeros(crop_band.shape, dtype=np.uint8)
    scaled = np.clip((crop_band[crop_valid] - p2) / denom * 255.0, 0, 255).astype(np.uint8)
    crop_u8[crop_valid] = scaled

    crop_norm = np.full(crop_band.shape, -1.0, dtype=np.float32)
    crop_norm[crop_valid] = (scaled.astype(np.float32) / 255.0 - 0.5) / 0.5

    return crop_norm, crop_u8, True, ymin, ymax, xmin, xmax, H, W


@torch.no_grad()
def predict_tiled(model, crop_norm, device, batch_size=32):
    """
    Performs tiled inference over the cropped region.
    Stitches probabilities using overlapping window averaging.
    """
    cH, cW = crop_norm.shape

    step = TILE_SIZE - OVERLAP
    ys = list(range(0, cH, step))
    xs = list(range(0, cW, step))

    prob_acc = np.zeros((cH, cW), dtype=np.float32)
    count_acc = np.zeros((cH, cW), dtype=np.float32)

    tiles_tensors = []
    tiles_info = []

    for y in ys:
        for x in xs:
            y_end = min(y + TILE_SIZE, cH)
            x_end = min(x + TILE_SIZE, cW)
            tile = crop_norm[y:y_end, x:x_end]

            # Skip tiles that are entirely background
            if np.all(tile == -1.0):
                continue

            th, tw = tile.shape
            if th < TILE_SIZE or tw < TILE_SIZE:
                pad_h = TILE_SIZE - th
                pad_w = TILE_SIZE - tw
                tile_padded = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='reflect')
            else:
                tile_padded = tile

            tiles_tensors.append(tile_padded[None, None])  # (1, 1, TILE, TILE)
            tiles_info.append((y, x, y_end, x_end, th, tw))

    if len(tiles_tensors) > 0:
        for i in range(0, len(tiles_tensors), batch_size):
            batch_arr = np.concatenate(tiles_tensors[i:i+batch_size], axis=0)
            inp = torch.from_numpy(batch_arr).to(device)
            logits = model(inp)
            preds = torch.sigmoid(logits).cpu().numpy()[:, 0]

            for idx in range(len(preds)):
                y, x, y_end, x_end, th, tw = tiles_info[i + idx]
                p_tile = preds[idx][:th, :tw]
                prob_acc[y:y_end, x:x_end] += p_tile
                count_acc[y:y_end, x:x_end] += 1.0

    count_acc = np.maximum(count_acc, 1e-6)
    crop_prob = prob_acc / count_acc
    return crop_prob


def make_overlay(sar_u8, binary_mask):
    """Creates a semi-transparent yellow/red overlay of oil pixels on grayscale SAR."""
    sar_rgb = cv2.cvtColor(sar_u8, cv2.COLOR_GRAY2BGR)
    overlay = sar_rgb.copy()
    oil = binary_mask > 0
    overlay[oil] = (0, 0, 255)  # Bright Red in BGR
    return cv2.addWeighted(sar_rgb, 0.6, overlay, 0.4, 0)


def process_scene(path, model, device):
    """Processes a single GeoTIFF scene, generating mask, overlay, and preview."""
    stem = os.path.splitext(os.path.basename(path))[0]
    t0 = time.time()

    with rasterio.open(path) as src:
        crs = src.crs
        transform = src.transform
        band = src.read(1).astype(np.float32)
        H, W = band.shape

    print(f"\nProcessing: {stem}", flush=True)
    print(f"  Raster dimensions: {W} x {H} | CRS: {crs}", flush=True)

    crop_norm, crop_u8, has_valid, ymin, ymax, xmin, xmax, H, W = preprocess_crop(band)

    binary_mask = np.zeros((H, W), dtype=np.uint8)
    sar_u8 = np.zeros((H, W), dtype=np.uint8)

    if not has_valid:
        print("  Warning: No valid SAR pixels found in scene.", flush=True)
    else:
        cH, cW = crop_norm.shape
        print(f"  Cropped active area: {cW} x {cH} (from full {W} x {H})", flush=True)
        crop_prob = predict_tiled(model, crop_norm, device)
        crop_binary = (crop_prob >= THRESHOLD).astype(np.uint8)

        binary_mask[ymin:ymax+1, xmin:xmax+1] = crop_binary
        sar_u8[ymin:ymax+1, xmin:xmax+1] = crop_u8

    # Calculate oil fraction relative to total scene area (H * W)
    total_pixels = H * W
    oil_pixel_count = int(np.count_nonzero(binary_mask > 0))
    oil_frac = float(oil_pixel_count) / float(total_pixels)
    spill_detected = oil_frac >= MIN_OIL_FRAC

    print(f"  Oil pixels: {oil_pixel_count:,} / {total_pixels:,} ({oil_frac:.6%})", flush=True)
    print(f"  Spill detected: {spill_detected} (threshold: {MIN_OIL_FRAC:.4%})", flush=True)

    # 1. Save Georeferenced Binary Mask GeoTIFF (MUST keep full dimensions, CRS & transform)
    mask_path = os.path.join(MASK_DIR, f"{stem}_mask.tif")
    with rasterio.open(
        mask_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
        compress="lzw"
    ) as dst:
        dst.write(binary_mask[None])

    # Downscale factors for PNG previews to optimize disk write speed
    max_side = max(H, W)
    scale_factor = 1.0
    if max_side > 3072:
        scale_factor = 3072.0 / float(max_side)
        out_w = int(W * scale_factor)
        out_h = int(H * scale_factor)
        sar_u8_disp = cv2.resize(sar_u8, (out_w, out_h), interpolation=cv2.INTER_AREA)
        binary_mask_disp = cv2.resize(binary_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    else:
        sar_u8_disp = sar_u8
        binary_mask_disp = binary_mask

    # 2. Save Overlay PNG (Fast PNG compression)
    overlay_img = make_overlay(sar_u8_disp, binary_mask_disp)
    overlay_path = os.path.join(OVERLAY_DIR, f"{stem}_overlay.png")
    cv2.imwrite(overlay_path, overlay_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    # 3. Save Preview PNG (SAR | Mask | Overlay)
    preview_path = os.path.join(PREVIEW_DIR, f"{stem}_preview.png")
    mask_vis = cv2.cvtColor((binary_mask_disp * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    sar_bgr = cv2.cvtColor(sar_u8_disp, cv2.COLOR_GRAY2BGR)

    disp_w = min(sar_u8_disp.shape[1], 768)
    disp_h = max(int(sar_u8_disp.shape[0] * disp_w / sar_u8_disp.shape[1]), 1)
    def resize_disp(img):
        return cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    side_by_side = np.concatenate([resize_disp(sar_bgr), resize_disp(mask_vis), resize_disp(overlay_img)], axis=1)
    cv2.imwrite(preview_path, side_by_side, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.2f}s", flush=True)

    return {
        "file": path,
        "spill_detected": bool(spill_detected),
        "oil_pixel_fraction": round(oil_frac, 8),
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "preview_path": preview_path,
        "processing_time_sec": round(elapsed, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Run U-Net inference on Sentinel-1 SAR GeoTIFF(s)")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to a single GeoTIFF file or directory of GeoTIFFs (defaults to data/arabian_sea_geotiff)")
    parser.add_argument("--image", type=str, default=None, help="Filter by image filename or substring (e.g. '20260421')")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional custom output directory")
    args = parser.parse_args()

    mask_dir = MASK_DIR
    overlay_dir = OVERLAY_DIR
    preview_dir = PREVIEW_DIR
    summary_file = SUMMARY_F

    if args.out_dir:
        mask_dir = os.path.join(args.out_dir, "masks")
        overlay_dir = os.path.join(args.out_dir, "overlays")
        preview_dir = os.path.join(args.out_dir, "previews")
        summary_file = os.path.join(args.out_dir, "summary.json")

    # Ensure output folders exist
    for folder in [mask_dir, overlay_dir, preview_dir]:
        os.makedirs(folder, exist_ok=True)

    if not os.path.isfile(CHECKPOINT):
        print(f"Error: Checkpoint file not found at '{CHECKPOINT}'", file=sys.stderr, flush=True)
        sys.exit(1)

    # 1. Resolve GeoTIFF files to process
    if args.input and os.path.isfile(args.input):
        tifs = [args.input]
        print(f"Single GeoTIFF target specified: '{args.input}'", flush=True)
    else:
        input_directory = args.input if (args.input and os.path.isdir(args.input)) else INPUT_DIR
        if not os.path.isdir(input_directory):
            print(f"Error: Input directory not found at '{input_directory}'", file=sys.stderr, flush=True)
            sys.exit(1)

        raw_files = sorted(
            glob.glob(os.path.join(input_directory, "*.tif")) +
            glob.glob(os.path.join(input_directory, "*.tiff"))
        )
        tifs = [f for f in raw_files if not os.path.basename(f).startswith("$(")]

        if args.image:
            img_kw = args.image.strip().lower()
            tifs = [f for f in tifs if img_kw in os.path.basename(f).lower()]
            print(f"Filtering by image keyword '{args.image}': {len(tifs)} match(es)", flush=True)
        else:
            print(f"Found {len(tifs)} GeoTIFF scene(s) in '{input_directory}'", flush=True)

    if len(tifs) == 0:
        print("No valid GeoTIFF files found to process.", flush=True)
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (CPU threads: {torch.get_num_threads()})", flush=True)

    # 2. Load model
    model = load_model(CHECKPOINT, device)

    # 3. Run inference per scene
    results = []
    for path in tqdm(tifs, desc="Processing Scenes"):
        try:
            res = process_scene(path, model, device)
            results.append(res)
        except Exception as e:
            print(f"  Error processing {os.path.basename(path)}: {e}", file=sys.stderr, flush=True)
            results.append({
                "file": path,
                "error": str(e),
                "spill_detected": False,
                "oil_pixel_fraction": 0.0
            })

    # 4. Save Summary JSON
    with open(SUMMARY_F, "w") as f:
        json.dump(results, f, indent=2)

    # 5. Final Report
    spill_count = sum(1 for r in results if r.get("spill_detected", False))
    print("\n" + "=" * 65, flush=True)
    print(" INFERENCE COMPLETE SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f" Total GeoTIFF scenes processed : {len(results)}", flush=True)
    print(f" Scenes with spill_detected=True : {spill_count} / {len(results)}", flush=True)
    print(f" Masks directory                 : {MASK_DIR}/", flush=True)
    print(f" Overlays directory              : {OVERLAY_DIR}/", flush=True)
    print(f" Previews directory              : {PREVIEW_DIR}/", flush=True)
    print(f" Summary JSON                    : {SUMMARY_F}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
