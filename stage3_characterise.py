"""
stage3_characterise.py — Stage 3: Spill Characterisation for SIH26143

Loads filtered GeoTIFF oil spill masks, extracts spatial features & georeferenced metrics
(pixel area, area in km², centroid lat/lon, perimeter in km, bounding box, orientation, major/minor axes),
and saves individual spill.json, spill_preview.png for each case, plus a combined stage3_summary.json.

Fully vectorised and optimized for instant processing of large GeoTIFF masks.

Next-step context:
Stage 3 outputs centroid + area for Stage 5 hindcast.
"""

import os
import sys
import glob
import json
import math
import shutil
import argparse
import numpy as np
import cv2
import rasterio
import pyproj
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def find_case_folders(base_dirs=None):
    """
    Finds case folders under 'cases/' or 'CASES/' that contain a filtered GeoTIFF mask.
    Deduplicates cases on Windows case-insensitive filesystems.
    Auto-detects any folder containing 'filtered_mask.tif' or '*_filtered_mask.tif'.
    """
    if base_dirs is None:
        candidates = ["cases", "CASES"]
        seen_base = set()
        base_dirs = []
        for b in candidates:
            if os.path.exists(b):
                real_b = os.path.normcase(os.path.realpath(b))
                if real_b not in seen_base:
                    seen_base.add(real_b)
                    base_dirs.append(b)

    case_dirs = []
    seen = set()

    for base in base_dirs:
        for entry in sorted(os.listdir(base)):
            full_path = os.path.join(base, entry)
            if not os.path.isdir(full_path):
                continue

            real_path = os.path.normcase(os.path.realpath(full_path))
            if real_path in seen:
                continue

            # Check for filtered_mask.tif or any *_filtered_mask.tif
            mask_candidates = sorted(
                glob.glob(os.path.join(full_path, "filtered_mask.tif")) +
                glob.glob(os.path.join(full_path, "*_filtered_mask.tif")) +
                glob.glob(os.path.join(full_path, "*filtered*.tif"))
            )

            # Filter out source SAR files if any matched
            mask_candidates = [m for m in mask_candidates if "Cal_Spk_TC_dB.tif" not in m or "filtered" in m]

            if mask_candidates:
                seen.add(real_path)
                case_dirs.append((entry, full_path, mask_candidates[0]))

    return case_dirs


def batch_transform_coords(transform, crs, rows, cols, transformer=None):
    """
    Transforms arrays of (rows, cols) pixel indices to (lons, lats) in WGS84 EPSG:4326.
    Uses fast vector matrix operations.
    """
    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)

    # Affine transform math with offset='center' (add 0.5)
    c_cols = cols + 0.5
    c_rows = rows + 0.5

    xs = transform.c + c_cols * transform.a + c_rows * transform.b
    ys = transform.f + c_cols * transform.d + c_rows * transform.e

    if crs and crs.is_projected:
        if transformer is None:
            transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(xs, ys)
    else:
        lons, lats = xs, ys

    return np.asarray(lons, dtype=np.float64), np.asarray(lats, dtype=np.float64)


def load_source_sar_u8(case_dir, target_h, target_w):
    """
    Loads source SAR GeoTIFF or overlay PNG from case directory,
    normalized to uint8 grayscale for visualization overlay.
    Reads decimated or from existing PNG to prevent memory exhaustion on large scenes.
    """
    # Priority 1: Use existing overlay PNG if present (instant, avoids multi-gigabyte GeoTIFF decodes)
    png_candidates = sorted(
        glob.glob(os.path.join(case_dir, "*overlay*.png")) +
        glob.glob(os.path.join(case_dir, "*.png"))
    )
    png_candidates = [p for p in png_candidates if "spill_preview" not in os.path.basename(p)]
    for p in png_candidates:
        try:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                if img.shape == (target_h, target_w):
                    return img
                return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        except Exception:
            pass

    # Priority 2: Read SAR GeoTIFF with rasterio decimation
    sar_candidates = sorted(
        glob.glob(os.path.join(case_dir, "*Cal_Spk_TC_dB.tif")) +
        glob.glob(os.path.join(case_dir, "source_geotiff.tif")) +
        glob.glob(os.path.join(case_dir, "*.tif"))
    )
    sar_candidates = [s for s in sar_candidates if "mask" not in os.path.basename(s).lower()]

    if sar_candidates:
        sar_path = sar_candidates[0]
        try:
            with rasterio.open(sar_path) as src:
                # Read directly decimated to target size to avoid loading 1.8GB into memory
                band = src.read(
                    1,
                    out_shape=(target_h, target_w),
                    resampling=rasterio.enums.Resampling.bilinear
                ).astype(np.float32)

            valid_mask = (band != 0) & (~np.isnan(band)) & np.isfinite(band)
            if np.any(valid_mask):
                valid_vals = band[valid_mask]
                p2, p98 = np.percentile(valid_vals, 2), np.percentile(valid_vals, 98)
                denom = max(float(p98 - p2), 1e-6)
                sar_u8 = np.zeros(band.shape, dtype=np.uint8)
                sar_u8[valid_mask] = np.clip((band[valid_mask] - p2) / denom * 255.0, 0, 255).astype(np.uint8)
                return sar_u8
        except Exception:
            pass

    # Fallback to dark background
    return np.full((target_h, target_w), 40, dtype=np.uint8)


def characterise_spill(case_id, case_dir, mask_path):
    """
    Performs Stage 3 spill characterisation on a single case folder mask.
    Computes georeferenced area, centroid lat/lon, perimeter, bbox, and orientation.
    """
    # Ensure filtered_mask.tif exists in case folder for standardized layout
    std_mask_path = os.path.join(case_dir, "filtered_mask.tif")
    if os.path.abspath(mask_path) != os.path.abspath(std_mask_path) and not os.path.exists(std_mask_path):
        try:
            shutil.copy2(mask_path, std_mask_path)
            mask_path = std_mask_path
        except Exception:
            pass

    with rasterio.open(mask_path) as src:
        crs = src.crs
        transform = src.transform
        mask = src.read(1).astype(np.uint8)
        H, W = mask.shape

    crs_str = str(crs) if crs else "EPSG:4326"

    # Ensure binary mask 0/1
    mask = (mask > 0).astype(np.uint8)
    area_pixels = int(np.count_nonzero(mask))

    # 1. Quality Check
    if area_pixels == 0:
        status = "empty_mask"
        results = {
            "case_id": case_id,
            "mask_path": mask_path.replace("\\", "/"),
            "status": status,
            "area_pixels": 0,
            "area_km2": 0.0,
            "centroid_lat": None,
            "centroid_lon": None,
            "perimeter_km": 0.0,
            "orientation_deg": None,
            "major_axis_km": None,
            "minor_axis_km": None,
            "bbox_pixels": None,
            "bbox_latlon": None,
            "crs": crs_str,
            "notes": "Stage 3 processed — empty mask (no oil pixels)"
        }

        preview = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(preview, f"Case: {case_id} (EMPTY MASK)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        preview_path = os.path.join(case_dir, "spill_preview.png")
        cv2.imwrite(preview_path, preview)

        spill_json_path = os.path.join(case_dir, "spill.json")
        with open(spill_json_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    status = "characterised"

    # 2. Centroid & Non-zero Coordinates Calculation
    rows, cols = np.where(mask > 0)
    centroid_row = float(np.mean(rows))
    centroid_col = float(np.mean(cols))

    # Pre-create transformer if projected
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if (crs and crs.is_projected) else None

    c_lons, c_lats = batch_transform_coords(transform, crs, [centroid_row], [centroid_col], transformer)
    centroid_lon = round(float(c_lons[0]), 6)
    centroid_lat = round(float(c_lats[0]), 6)

    # 3. Area Calculation (m² → km²)
    geod = pyproj.Geod(ellps="WGS84")
    if crs and crs.is_projected:
        pixel_w_m = abs(transform.a)
        pixel_h_m = abs(transform.e)
        pixel_area_m2 = pixel_w_m * pixel_h_m
    else:
        deg_x = abs(transform.a)
        deg_y = abs(transform.e)
        _, _, w_m = geod.inv(centroid_lon - deg_x/2.0, centroid_lat, centroid_lon + deg_x/2.0, centroid_lat)
        _, _, h_m = geod.inv(centroid_lon, centroid_lat - deg_y/2.0, centroid_lon, centroid_lat + deg_y/2.0)
        pixel_area_m2 = abs(w_m * h_m)

    total_area_m2 = area_pixels * pixel_area_m2
    area_km2 = round(total_area_m2 / 1.0e6, 4)

    # 4. Bounding Box Calculation
    min_row, max_row = int(np.min(rows)), int(np.max(rows))
    min_col, max_col = int(np.min(cols)), int(np.max(cols))
    bbox_pixels = [min_row, min_col, max_row, max_col]

    corner_rows = [min_row, min_row, max_row, max_row]
    corner_cols = [min_col, max_col, min_col, max_col]
    cb_lons, cb_lats = batch_transform_coords(transform, crs, corner_rows, corner_cols, transformer)

    bbox_latlon = [
        round(float(np.min(cb_lons)), 6),
        round(float(np.min(cb_lats)), 6),
        round(float(np.max(cb_lons)), 6),
        round(float(np.max(cb_lats)), 6)
    ]

    # 5. Crop Active Bounding Box for High-Speed Memory-Safe OpenCV Processing
    pad = 32
    r0, r1 = max(0, min_row - pad), min(H, max_row + pad + 1)
    c0, c1 = max(0, min_col - pad), min(W, max_col + pad + 1)
    cropped_mask = mask[r0:r1, c0:c1]

    # Connected Components Analysis on cropped bounding box
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cropped_mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(areas))
        largest_mask = (labels == largest_idx).astype(np.uint8)
    else:
        largest_mask = cropped_mask.copy()

    # Perimeter Calculation (Fast vector contour arcLength on cropped region)
    contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    px_scale_m = math.sqrt(pixel_area_m2)
    total_perimeter_m = float(sum(cv2.arcLength(c, True) for c in contours) * px_scale_m)
    perimeter_km = round(total_perimeter_m / 1000.0, 4)

    # 6. Orientation & Major/Minor Axes (fitEllipse on largest component)
    blob_pts = np.argwhere(largest_mask > 0)
    if len(blob_pts) >= 5:
        # Subsample if point cloud is large for instant fitting
        if len(blob_pts) > 10000:
            step = len(blob_pts) // 10000
            sampled_pts = blob_pts[::step]
        else:
            sampled_pts = blob_pts

        blob_cv = np.flip(sampled_pts, axis=1).astype(np.int32).reshape(-1, 1, 2)
        try:
            (cx_e, cy_e), (d1, d2), angle = cv2.fitEllipse(blob_cv)
            orientation_deg = round(float(angle), 2)
            major_axis_px = float(max(d1, d2))
            minor_axis_px = float(min(d1, d2))

            px_scale_km = math.sqrt(pixel_area_m2) / 1000.0
            major_axis_km = round(major_axis_px * px_scale_km, 4)
            minor_axis_km = round(minor_axis_px * px_scale_km, 4)
        except Exception:
            orientation_deg = None
            major_axis_km = None
            minor_axis_km = None
    else:
        orientation_deg = None
        major_axis_km = None
        minor_axis_km = None

    # 8. Create spill_preview.png (pre-downsampled to max 2048 to prevent multi-gigabyte memory allocations)
    max_dim = max(H, W)
    if max_dim > 2048:
        scale = 2048.0 / float(max_dim)
        preview_w, preview_h = max(int(W * scale), 1), max(int(H * scale), 1)
    else:
        scale = 1.0
        preview_w, preview_h = W, H

    if scale < 1.0:
        mask_preview = cv2.resize(mask, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_preview = mask

    sar_u8 = load_source_sar_u8(case_dir, preview_h, preview_w)
    sar_bgr = cv2.cvtColor(sar_u8, cv2.COLOR_GRAY2BGR)

    overlay = sar_bgr.copy()
    overlay[mask_preview > 0] = (0, 0, 255)  # BGR Red
    disp_out = cv2.addWeighted(sar_bgr, 0.5, overlay, 0.5, 0)

    # Draw bounding box scaled to preview
    p_min_col, p_min_row = int(round(min_col * scale)), int(round(min_row * scale))
    p_max_col, p_max_row = int(round(max_col * scale)), int(round(max_row * scale))
    cv2.rectangle(disp_out, (p_min_col, p_min_row), (p_max_col, p_max_row), (0, 255, 255), 2)

    # Draw centroid marker scaled to preview
    cr_col, cr_row = int(round(centroid_col * scale)), int(round(centroid_row * scale))
    cv2.circle(disp_out, (cr_col, cr_row), 12, (0, 255, 255), 2)
    cv2.circle(disp_out, (cr_col, cr_row), 3, (0, 255, 255), -1)
    cv2.line(disp_out, (cr_col - 20, cr_row), (cr_col + 20, cr_row), (0, 255, 255), 2)
    cv2.line(disp_out, (cr_col, cr_row - 20), (cr_col, cr_row + 20), (0, 255, 255), 2)


    # Draw text information banner on preview image
    banner_h = 75
    banner = np.zeros((banner_h, disp_out.shape[1], 3), dtype=np.uint8)
    line1 = f"Case: {case_id}  |  Area: {area_km2} km2 ({area_pixels:,} px)  |  Status: {status}"
    line2 = f"Centroid: {centroid_lat:.6f} N, {centroid_lon:.6f} E  |  Perimeter: {perimeter_km} km"
    if orientation_deg is not None:
        line2 += f"  |  Orient: {orientation_deg:.1f} deg"

    cv2.putText(banner, line1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(banner, line2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

    final_preview = np.vstack([banner, disp_out])
    preview_path = os.path.join(case_dir, "spill_preview.png")
    cv2.imwrite(preview_path, final_preview, [cv2.IMWRITE_PNG_COMPRESSION, 2])

    # 9. Format spill.json
    results = {
        "case_id": case_id,
        "mask_path": mask_path.replace("\\", "/"),
        "status": status,
        "area_pixels": area_pixels,
        "area_km2": area_km2,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "perimeter_km": perimeter_km,
        "orientation_deg": orientation_deg,
        "major_axis_km": major_axis_km,
        "minor_axis_km": minor_axis_km,
        "bbox_pixels": bbox_pixels,
        "bbox_latlon": bbox_latlon,
        "crs": crs_str,
        "notes": "Stage 3 from filtered U-Net mask"
    }

    spill_json_path = os.path.join(case_dir, "spill.json")
    with open(spill_json_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="SIH26143 — STAGE 3: SPILL CHARACTERISATION")
    parser.add_argument("--case", type=str, default=None, help="Target case ID (e.g. CASE_a1 or all)")
    args = parser.parse_args()

    print("=" * 85, flush=True)
    print(" SIH26143 — STAGE 3: SPILL CHARACTERISATION", flush=True)
    print("=" * 85, flush=True)

    case_dirs = find_case_folders()
    if not case_dirs:
        print("Error: No case folders with filtered_mask.tif found under 'cases/' or 'CASES/'.", flush=True)
        sys.exit(1)

    if args.case and args.case.lower() != "all":
        t_norm = args.case.lower().replace("_", " ").strip()
        case_dirs = [c for c in case_dirs if c[0].lower().replace("_", " ").strip() == t_norm or c[0].lower() == args.case.lower()]
        if not case_dirs:
            print(f"Error: Target case '{args.case}' not found.", flush=True)
            sys.exit(1)

    print(f"Found {len(case_dirs)} case folder(s) to process:", flush=True)
    for cid, cdir, mpath in case_dirs:
        print(f"  - [{cid}] in '{cdir}' using mask '{os.path.basename(mpath)}'", flush=True)
    print("-" * 85, flush=True)

    summary = []
    for case_id, case_dir, mask_path in tqdm(case_dirs, desc="Characterising Spills"):
        try:
            res = characterise_spill(case_id, case_dir, mask_path)
            summary.append(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nError processing case '{case_id}': {e}", file=sys.stderr, flush=True)
            sys.exit(1)

    # Save combined stage3_summary.json in cases/ root
    cases_root = "cases" if os.path.exists("cases") else "CASES"
    summary_path = os.path.join(cases_root, "stage3_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 85, flush=True)
    print(" STAGE 3 CHARACTERISATION SUMMARY TABLE", flush=True)
    print("=" * 85, flush=True)
    header = f"{'case_id':<18} | {'area_km2':<12} | {'lat':<12} | {'lon':<12} | {'status':<15}"
    print(header, flush=True)
    print("-" * 85, flush=True)
    for s in summary:
        cid = s["case_id"][:17]
        akm = f"{s['area_km2']:.4f}" if s["area_km2"] is not None else "N/A"
        lat = f"{s['centroid_lat']:.6f}" if s["centroid_lat"] is not None else "N/A"
        lon = f"{s['centroid_lon']:.6f}" if s["centroid_lon"] is not None else "N/A"
        st = s["status"]
        print(f"{cid:<18} | {akm:<12} | {lat:<12} | {lon:<12} | {st:<15}", flush=True)
    print("=" * 85, flush=True)
    print(f"Summary JSON saved to: {summary_path}", flush=True)
    print("=" * 85, flush=True)

    print("\nStage 3 outputs centroid + area for Stage 5 hindcast.", flush=True)


if __name__ == "__main__":
    main()
