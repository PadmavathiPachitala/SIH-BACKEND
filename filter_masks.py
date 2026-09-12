"""
filter_masks.py — Post-U-Net Filter Layer for Arabian Sea Oil Spill Detection.

Filters raw U-Net binary masks to remove small noise blobs, edge artifacts, and
unreliable high-area noise scenes. Computes confidence scores and classifies each scene as:
- 'high-confidence_candidate'
- 'no_confident_spill'
"""

import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# ── TUNABLE CONSTANTS ────────────────────────────────────────────────────────
MIN_PIXELS      = 500      # Drop connected components smaller than 500 pixels
BORDER_WIDTH    = 5        # Margin (pixels) to check border touch
EDGE_TOUCH_FRAC = 0.30     # Drop component if >30% of pixels touch border
MAX_OIL_FRAC    = 0.05     # Reject scene if remaining oil > 5% of scene
CONF_THRESHOLD  = 0.55     # Minimum confidence to mark as high-confidence candidate
# ─────────────────────────────────────────────────────────────────────────────

RAW_MASK_DIR    = os.path.join("inference_outputs", "masks")
SAR_DIR         = os.path.join("data", "arabian_sea_geotiff")

OUT_ROOT        = "inference_outputs"
FILTER_MASK_DIR = os.path.join(OUT_ROOT, "filtered_masks")
OVERLAY_DIR     = os.path.join(OUT_ROOT, "filtered_overlays")
SUMMARY_F       = os.path.join(OUT_ROOT, "filtered_summary.json")


def load_sar_u8(sar_path, target_h, target_w, ymin=0, ymax=None, xmin=0, xmax=None):
    """Loads SAR GeoTIFF, normalizes dB to uint8 [0, 255] for visual overlay."""
    if not os.path.isfile(sar_path):
        return np.zeros((target_h, target_w), dtype=np.uint8)

    try:
        with rasterio.open(sar_path) as src:
            if ymax is not None and xmax is not None:
                win_h = max(ymax - ymin + 1, 1)
                win_w = max(xmax - xmin + 1, 1)
                window = Window(xmin, ymin, win_w, win_h)
                band = src.read(1, window=window).astype(np.float32)
            else:
                band = src.read(1).astype(np.float32)

        valid_mask = (band != 0) & (~np.isnan(band)) & np.isfinite(band)
        if not np.any(valid_mask):
            return np.zeros(band.shape, dtype=np.uint8)

        valid_vals = band[valid_mask]
        p2, p98 = np.percentile(valid_vals, 2), np.percentile(valid_vals, 98)
        denom = max(float(p98 - p2), 1e-6)

        sar_u8 = np.zeros(band.shape, dtype=np.uint8)
        sar_u8[valid_mask] = np.clip((band[valid_mask] - p2) / denom * 255.0, 0, 255).astype(np.uint8)
        return sar_u8
    except Exception as e:
        print(f"Warning: Failed to load SAR overlay {sar_path}: {e}", file=sys.stderr, flush=True)
        return np.zeros((target_h, target_w), dtype=np.uint8)


def make_filtered_overlay(sar_u8, filtered_mask):
    """Generates visual PNG with remaining filtered oil pixels in red."""
    sar_bgr = cv2.cvtColor(sar_u8, cv2.COLOR_GRAY2BGR)
    overlay = sar_bgr.copy()
    oil_mask = filtered_mask > 0
    overlay[oil_mask] = (0, 0, 255)  # Bright Red in BGR
    return cv2.addWeighted(sar_bgr, 0.6, overlay, 0.4, 0)


def process_mask(mask_path):
    """Processes a single raw mask TIFF through component, border, area & confidence filters."""
    stem = os.path.basename(mask_path).replace("_mask.tif", "")
    sar_path = os.path.join(SAR_DIR, f"{stem}.tif")
    if not os.path.isfile(sar_path):
        sar_path = os.path.join(SAR_DIR, f"{stem}.tiff")

    t0 = time.time()
    with rasterio.open(mask_path) as src:
        crs = src.crs
        transform = src.transform
        mask = src.read(1).astype(np.uint8)
        H, W = mask.shape

    # Ensure binary 0/1
    mask = (mask > 0).astype(np.uint8)
    raw_pixels = int(np.count_nonzero(mask))

    if raw_pixels == 0:
        filtered_mask = np.zeros((H, W), dtype=np.uint8)
        status = "no_confident_spill"
        reject_reasons = ["empty_raw_mask"]
        confidence = 0.0
        filtered_pixels = 0
        filtered_frac = 0.0
        ymin, ymax, xmin, xmax = 0, H - 1, 0, W - 1
    else:
        # Crop active bounding box to optimize performance
        rows, cols = np.where(mask > 0)
        ymin, ymax = int(rows.min()), int(rows.max())
        xmin, xmax = int(cols.min()), int(cols.max())
        crop_mask = mask[ymin:ymax+1, xmin:xmax+1]

        # 3a. Connected Components Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(crop_mask, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]

        # Size filter (< MIN_PIXELS)
        valid_ids = np.where(areas >= MIN_PIXELS)[0]
        valid_ids = [idx for idx in valid_ids if idx > 0]  # Exclude background (0)
        dropped_small_count = (num_labels - 1) - len(valid_ids)

        # 3b. Edge Filter (border touch > EDGE_TOUCH_FRAC)
        final_ids = []
        dropped_edge_count = 0
        valid_areas = []
        valid_centroids = []

        for i in valid_ids:
            area = int(stats[i, cv2.CC_STAT_AREA])
            c_left = int(stats[i, cv2.CC_STAT_LEFT])
            c_top = int(stats[i, cv2.CC_STAT_TOP])
            c_w = int(stats[i, cv2.CC_STAT_WIDTH])
            c_h = int(stats[i, cv2.CC_STAT_HEIGHT])

            g_left = xmin + c_left
            g_top = ymin + c_top
            g_right = g_left + c_w
            g_bottom = g_top + c_h

            is_near_border = (g_top < BORDER_WIDTH) or (g_bottom >= H - BORDER_WIDTH) or (g_left < BORDER_WIDTH) or (g_right >= W - BORDER_WIDTH)

            if is_near_border:
                sub_label = (labels[c_top:c_top+c_h, c_left:c_left+c_w] == i)
                sub_r, sub_c = np.where(sub_label)
                sub_g_r = ymin + c_top + sub_r
                sub_g_c = xmin + c_left + sub_c
                near_b = (sub_g_r < BORDER_WIDTH) | (sub_g_r >= H - BORDER_WIDTH) | (sub_g_c < BORDER_WIDTH) | (sub_g_c >= W - BORDER_WIDTH)
                border_touch_frac = float(np.count_nonzero(near_b)) / float(area)

                if border_touch_frac > EDGE_TOUCH_FRAC:
                    dropped_edge_count += 1
                    continue

            final_ids.append(i)
            valid_areas.append(area)
            cy = float(centroids[i][1]) + ymin
            cx = float(centroids[i][0]) + xmin
            valid_centroids.append((cy, cx))

        filtered_mask = np.zeros((H, W), dtype=np.uint8)
        if len(final_ids) > 0:
            filtered_crop = np.isin(labels, final_ids).astype(np.uint8)
            filtered_mask[ymin:ymax+1, xmin:xmax+1] = filtered_crop
            filtered_pixels = int(np.count_nonzero(filtered_crop))
        else:
            filtered_pixels = 0

        filtered_frac = float(filtered_pixels) / float(H * W)
        reject_reasons = []

        if dropped_small_count > 0 and len(final_ids) == 0:
            reject_reasons.append("too_small")
        if dropped_edge_count > 0 and len(final_ids) == 0:
            reject_reasons.append("edge_only")
        if filtered_pixels == 0 and not reject_reasons:
            reject_reasons.append("empty_after_filter")

        # 3c. Max-area filter (> MAX_OIL_FRAC)
        if filtered_frac > MAX_OIL_FRAC:
            reject_reasons.append("too_large_fraction")

        # 4. Compute Confidence Score (0.0 to 1.0)
        if filtered_pixels > 0 and len(valid_areas) > 0:
            max_area = max(valid_areas)
            s_compact = max_area / float(filtered_pixels)  # Single dominant spill vs scattered specks
            s_frac = max(0.0, 1.0 - (filtered_frac / MAX_OIL_FRAC))

            c_dists = []
            for cy, cx in valid_centroids:
                dy = min(cy, H - cy) / (H / 2.0)
                dx = min(cx, W - cx) / (W / 2.0)
                c_dists.append(min(dy, dx))
            s_dist = float(np.mean(c_dists))

            confidence = round(0.40 * s_compact + 0.35 * s_frac + 0.25 * s_dist, 4)
            confidence = float(np.clip(confidence, 0.0, 1.0))
        else:
            confidence = 0.0

        if filtered_pixels > 0 and confidence < CONF_THRESHOLD:
            reject_reasons.append("low_confidence")

        # 5. Final Status Decision
        if filtered_pixels > 0 and confidence >= CONF_THRESHOLD and "too_large_fraction" not in reject_reasons:
            status = "high-confidence_candidate"
        else:
            status = "no_confident_spill"

    # 6. Save Filtered GeoTIFF Mask (Preserving CRS and Affine Transform)
    filtered_mask_path = os.path.join(FILTER_MASK_DIR, f"{stem}_filtered_mask.tif")
    with rasterio.open(
        filtered_mask_path, "w",
        driver="GTiff",
        height=H, width=W,
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
        compress="lzw"
    ) as dst:
        dst.write(filtered_mask[None])

    # 6b. Save Filtered Overlay PNG (Cropped active region for fast PNG export)
    crop_h = ymax - ymin + 1
    crop_w = xmax - xmin + 1
    sar_crop_u8 = load_sar_u8(sar_path, crop_h, crop_w, ymin, ymax, xmin, xmax)
    filt_crop = filtered_mask[ymin:ymax+1, xmin:xmax+1]

    max_side = max(crop_h, crop_w)
    if max_side > 2048:
        scale = 2048.0 / float(max_side)
        out_w, out_h = max(int(crop_w * scale), 1), max(int(crop_h * scale), 1)
        sar_disp = cv2.resize(sar_crop_u8, (out_w, out_h), interpolation=cv2.INTER_AREA)
        filt_disp = cv2.resize(filt_crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    else:
        sar_disp = sar_crop_u8
        filt_disp = filt_crop

    overlay_img = make_filtered_overlay(sar_disp, filt_disp)
    overlay_path = os.path.join(OVERLAY_DIR, f"{stem}_filtered_overlay.png")
    cv2.imwrite(overlay_path, overlay_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    elapsed = time.time() - t0

    return {
        "file": sar_path if os.path.isfile(sar_path) else mask_path,
        "raw_mask_path": mask_path,
        "status": status,
        "confidence": confidence,
        "oil_pixels_raw": raw_pixels,
        "oil_pixels_filtered": filtered_pixels,
        "oil_fraction_filtered": round(filtered_frac, 8),
        "reject_reasons": reject_reasons,
        "filtered_mask_path": filtered_mask_path,
        "filtered_overlay_path": overlay_path,
        "processing_time_sec": round(elapsed, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Post-U-Net Filter Layer for Arabian Sea Oil Spill Detection")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to a single raw mask .tif or directory of raw masks (defaults to inference_outputs/masks)")
    parser.add_argument("--image", "--mask", type=str, default=None, help="Filter by scene/mask filename or substring (e.g. '20260421')")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional custom output directory")
    args = parser.parse_args()

    filter_mask_dir = FILTER_MASK_DIR
    overlay_dir = OVERLAY_DIR
    summary_file = SUMMARY_F

    if args.out_dir:
        filter_mask_dir = os.path.join(args.out_dir, "filtered_masks")
        overlay_dir = os.path.join(args.out_dir, "filtered_overlays")
        summary_file = os.path.join(args.out_dir, "filtered_summary.json")

    # Ensure output folders exist
    for folder in [filter_mask_dir, overlay_dir]:
        os.makedirs(folder, exist_ok=True)

    # 1. Resolve raw mask files to process
    if args.input and os.path.isfile(args.input):
        raw_masks = [args.input]
        print(f"Single raw mask target specified: '{args.input}'", flush=True)
    else:
        raw_mask_dir = args.input if (args.input and os.path.isdir(args.input)) else RAW_MASK_DIR
        if not os.path.isdir(raw_mask_dir):
            print(f"Error: Raw mask directory not found at '{raw_mask_dir}'", file=sys.stderr, flush=True)
            sys.exit(1)

        raw_masks = sorted(
            glob.glob(os.path.join(raw_mask_dir, "*.tif")) +
            glob.glob(os.path.join(raw_mask_dir, "*.tiff"))
        )

        if args.image:
            kw = args.image.strip().lower()
            raw_masks = [m for m in raw_masks if kw in os.path.basename(m).lower()]
            print(f"Filtering by mask keyword '{args.image}': {len(raw_masks)} match(es)", flush=True)
        else:
            print(f"Found {len(raw_masks)} raw GeoTIFF mask(s) in '{raw_mask_dir}'", flush=True)

    if len(raw_masks) == 0:
        print("No raw mask files found to filter.", flush=True)
        sys.exit(0)

    print("\nStarting Post-U-Net Filtering Layer...", flush=True)
    print(f"  MIN_PIXELS      = {MIN_PIXELS}")
    print(f"  BORDER_WIDTH    = {BORDER_WIDTH}")
    print(f"  EDGE_TOUCH_FRAC = {EDGE_TOUCH_FRAC}")
    print(f"  MAX_OIL_FRAC    = {MAX_OIL_FRAC}")
    print(f"  CONF_THRESHOLD  = {CONF_THRESHOLD}\n", flush=True)

    results = []
    for mask_path in tqdm(raw_masks, desc="Filtering Masks"):
        try:
            res = process_mask(mask_path)
            results.append(res)
        except Exception as e:
            print(f"  Error filtering {os.path.basename(mask_path)}: {e}", file=sys.stderr, flush=True)
            results.append({
                "file": mask_path,
                "status": "no_confident_spill",
                "confidence": 0.0,
                "oil_pixels_raw": 0,
                "oil_pixels_filtered": 0,
                "oil_fraction_filtered": 0.0,
                "reject_reasons": [f"processing_error: {str(e)}"],
                "filtered_mask_path": "",
                "filtered_overlay_path": ""
            })

    # Save summary JSON
    with open(SUMMARY_F, "w") as f:
        json.dump(results, f, indent=2)

    # Print Summary Table
    cand_count = sum(1 for r in results if r["status"] == "high-confidence_candidate")
    rej_count = sum(1 for r in results if r["status"] == "no_confident_spill")

    print("\n" + "=" * 75, flush=True)
    print(" FILTER LAYER CLASSIFICATION SUMMARY", flush=True)
    print("=" * 75, flush=True)
    print(f"{'Stem':<42} | {'Status':<25} | {'Conf':<6} | {'Filt Pixels':<10}", flush=True)
    print("-" * 75, flush=True)
    for r in results:
        stem = os.path.basename(r["raw_mask_path"]).replace("_mask.tif", "")[:40]
        status = r["status"]
        conf = r["confidence"]
        fp = r["oil_pixels_filtered"]
        print(f"{stem:<42} | {status:<25} | {conf:<6.4f} | {fp:<10,d}", flush=True)
    print("=" * 75, flush=True)
    print(f" High-Confidence Candidates : {cand_count} / {len(results)}", flush=True)
    print(f" No Confident Spill         : {rej_count} / {len(results)}", flush=True)
    print(f" Filtered Masks directory   : {FILTER_MASK_DIR}/", flush=True)
    print(f" Filtered Overlays dir      : {OVERLAY_DIR}/", flush=True)
    print(f" Filtered Summary JSON      : {SUMMARY_F}", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
