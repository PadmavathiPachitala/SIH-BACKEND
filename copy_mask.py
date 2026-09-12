"""
copy_mask.py — Accurately copies the 2026-08-02 filtered mask into CASE a3.
"""
import os
import shutil
import glob

def main():
    src_pattern = os.path.join("inference_outputs", "filtered_masks", "*20260802*filtered_mask.tif")
    matches = glob.glob(src_pattern)
    if not matches:
        print(f"Error: No source mask matching '{src_pattern}' found.")
        return

    src = matches[0]
    src_size = os.path.getsize(src)
    print(f"Found source mask: {src} ({src_size:,} bytes)")

    for target_base in ["CASES", "cases"]:
        if os.path.exists(target_base):
            target_dir = os.path.join(target_base, "CASE a3")
            os.makedirs(target_dir, exist_ok=True)
            dst = os.path.join(target_dir, "filtered_mask.tif")
            shutil.copy2(src, dst)
            dst_size = os.path.getsize(dst)
            print(f"Successfully copied to: {dst} ({dst_size:,} bytes)")

if __name__ == "__main__":
    main()
