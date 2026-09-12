"""
process_new_tif.py — Universal All-in-One Processor for Any New Satellite GeoTIFF.

Automates the entire end-to-end flow in a single command:
1. U-Net Deep Semantic Segmentation (infer_geotiff.py)
2. Swath Edge & Morphological Filtering (filter_masks.py)
3. Case Directory Setup & Standardized Mask Placement
4. Full 8-Stage Pipeline (Spill Geometry, Age, Hindcast, 96h Forecast, AIS, Suspect Ranking)
5. Master JSON dossier generation in CASES/<case>/ and my_dashboard_data/ for frontend upload.

Usage:
  python process_new_tif.py --image "path/to/your_satellite_image.tif" --case "CASE a4"
"""

import os
import sys
import shutil
import argparse
import subprocess

def run_cmd(cmd_list, description):
    print(f"\n>>> [RUNNING] {description}...", flush=True)
    res = subprocess.run(cmd_list, text=True)
    if res.returncode != 0:
        print(f"\n[FAIL] {description} failed with return code {res.returncode}.", file=sys.stderr)
        sys.exit(res.returncode)
    print(f">>> [DONE] {description} completed successfully.\n", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Process any new satellite GeoTIFF into the full oil spill pipeline and dashboard.")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to the new Sentinel-1 SAR GeoTIFF (.tif) image")
    parser.add_argument("--case", "-c", type=str, required=True, help="Case identifier (e.g. 'CASE a4', 'CASE a5')")
    parser.add_argument("--force", action="store_true", help="Force re-run all pipeline stages")
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        print(f"Error: Specified image file not found at '{image_path}'", file=sys.stderr)
        sys.exit(1)

    py_exe = sys.executable
    case_name = args.case.strip()
    case_id_norm = case_name.replace(" ", "_")
    base_stem = os.path.splitext(os.path.basename(image_path))[0]

    print("=" * 80)
    print(f" SIH26143 — AUTOMATED NEW GEOTIFF INGESTION PIPELINE")
    print(f" Target Image : {image_path}")
    print(f" Target Case  : {case_name}")
    print("=" * 80)

    # 1. Run U-Net Segmentation on this single image
    run_cmd([py_exe, "infer_geotiff.py", "--input", image_path], "Step 1: U-Net SAR Segmentation")

    # 2. Locate generated raw mask
    raw_mask_path = os.path.join("inference_outputs", "masks", f"{base_stem}_mask.tif")
    if not os.path.isfile(raw_mask_path):
        print(f"Error: Expected raw mask not found at '{raw_mask_path}'", file=sys.stderr)
        sys.exit(1)

    # 3. Run Post-U-Net Filtering Layer
    run_cmd([py_exe, "filter_masks.py", "--input", raw_mask_path], "Step 2: Post-U-Net Swath & Noise Filtering")

    # 4. Locate generated filtered mask
    filt_mask_path = os.path.join("inference_outputs", "filtered_masks", f"{base_stem}_filtered_mask.tif")
    if not os.path.isfile(filt_mask_path):
        print(f"Error: Expected filtered mask not found at '{filt_mask_path}'", file=sys.stderr)
        sys.exit(1)

    # 5. Set up Case Folders in CASES/ and cases/
    for base_dir in ["CASES", "cases"]:
        c_dir = os.path.join(base_dir, case_name)
        os.makedirs(c_dir, exist_ok=True)
        dest_mask = os.path.join(c_dir, "filtered_mask.tif")
        shutil.copy2(filt_mask_path, dest_mask)
        print(f"Copied filtered mask -> {dest_mask} ({os.path.getsize(dest_mask):,} bytes)")

    # 6. Run the End-to-End Pipeline
    pipeline_args = [py_exe, "run_pipeline.py", "--case", case_name]
    if args.force:
        pipeline_args.append("--force")
    run_cmd(pipeline_args, f"Step 3: End-to-End Pipeline for [{case_name}]")

    # 7. Merge into Master Dossier and Frontend Export
    case_target_dir = os.path.join("CASES", case_name)
    if not os.path.exists(case_target_dir):
        case_target_dir = os.path.join("cases", case_name)
    run_cmd([py_exe, "merge_case_json.py", "--case_dir", case_target_dir], "Step 4: Merging Dossier & Exporting to Dashboard")

    clean_name = case_name.lower().replace(" ", "_")
    frontend_json = os.path.join("my_dashboard_data", f"{clean_name}.json")

    print("\n" + "=" * 80)
    print(f" ALL PROCESSING COMPLETED SUCCESSFULLY FOR [{case_name}]!")
    print("=" * 80)
    print(f" - Case Directory  : {case_target_dir}")
    print(f" - Master Dossier  : {os.path.join(case_target_dir, 'case_full.json')}")
    print(f" - Frontend Upload : {frontend_json}")
    print(f" - Web App View    : Refresh http://localhost:5000 and select '{case_name}'")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
