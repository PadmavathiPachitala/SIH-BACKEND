# 🚀 EXECUTE: Processing a New Satellite GeoTIFF (.tif) Image

This guide explains how to process any new Sentinel-1 SAR satellite image (`.tif`), run the complete 8-stage oil spill detection & attribution pipeline, and export the output into `my_dashboard_data/` for frontend upload.

---

## ⚡ Option 1: One-Click Execution (Recommended)

You can run the entire process in **one single command** using `process_new_tif.py` or `execute.bat`:

### Using Python:
```bash
python process_new_tif.py --image "path\to\your_image.tif" --case "CASE a4" --force
```

### Using Windows Batch File:
```cmd
execute.bat "path\to\your_image.tif" "CASE a4"
```

---

## 🛠 Option 2: Step-by-Step Execution

If you prefer to run each stage manually:

### Step 1: Deep Learning U-Net Semantic Segmentation
Run inference on your single GeoTIFF file:
```bash
python infer_geotiff.py --input "path\to\your_image.tif"
```
* **Output:** `inference_outputs\masks\<image_name>_mask.tif`

---

### Step 2: Swath Edge & Morphological Noise Filtering
Filter the raw segmentation mask to eliminate border artifacts and small specks:
```bash
python filter_masks.py --input "inference_outputs\masks\<image_name>_mask.tif"
```
* **Output:** `inference_outputs\filtered_masks\<image_name>_filtered_mask.tif`

---

### Step 3: Setup Case Directory
Create the target case folder and place the filtered mask:
```cmd
mkdir "CASES\CASE a4"
copy "inference_outputs\filtered_masks\<image_name>_filtered_mask.tif" "CASES\CASE a4\filtered_mask.tif"
```

---

### Step 4: Run the End-to-End 8-Stage Pipeline
Execute the full physics, ocean current, and vessel attribution stages:
```bash
python run_pipeline.py --case CASE_a4 --force
```

This runs:
- **Stage 3:** Spill Characterisation (Area in km², Centroid Lat/Lon, Perimeter, Ellipse orientation)
- **Stage 3b:** Oil Slick Age Estimation (Evaporation & spreading physics)
- **Stage 5:** Lagrangian Ocean Current Hindcasting (Discharge origin backtracking)
- **Stage 5b:** 24h / 72h / 96h Forward Trajectory & Probability Envelope Forecasting
- **Stage 6:** AIS Vessel Spatio-Temporal Gating
- **Stage 7:** MCDA Multi-Factor Suspect Vessel Attribution Ranking

---

### Step 5: Merge Dossier & Export to Frontend
Compile the outputs into master format:
```bash
python merge_case_json.py --case_dir "CASES/CASE a4"
```

This automatically generates:
1. `CASES/CASE a4/case_full.json` (Local dossier)
2. `my_dashboard_data/case_a4.json` (**Ready for direct upload to your frontend!**)

---

### Step 6: View in the Web Dashboard
1. Run the local dashboard:
   ```bash
   python app.py
   ```
2. Open your browser at:
   ```
   http://localhost:5000
   ```
3. Select your new case from the dropdown menu at the top.

---

## 📁 Key File Locations

| Output | Path | Description |
| :--- | :--- | :--- |
| **Mask** | `CASES/<case>/filtered_mask.tif` | Clean binary spill raster |
| **Geometry** | `CASES/<case>/spill.json` | Area, centroid, coordinates |
| **Hindcast** | `CASES/<case>/stage5/hindcast.json` | Spill origin coordinates |
| **Forecast** | `CASES/<case>/forecast_envelope.geojson` | 50% & 90% drift cones |
| **Suspects** | `CASES/<case>/stage7/ranked_suspects.json` | Top ranked vessels & scores |
| **Frontend Upload** | `my_dashboard_data/<case>.json` | Complete master JSON for frontend |
