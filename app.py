#!/usr/bin/env python3
"""
=====================================================================================
 SIH26143 — ARABIAN SEA OIL SPILL DETECTION & ATTRIBUTION SYSTEM
 Localhost Interactive Web Application & Slide Presentation Dashboard
=====================================================================================
 Theme: Ultra-Clean High-Contrast White & Black (with 1-Click Dark/Light Toggle)
 Features:
   - Dual-Mode White & Black Theme:
       * Mode 1: Crisp Paper White & Jet Black (Stark Minimalist Editorial)
       * Mode 2: Deep Obsidian Black & Pure White (High-Contrast Monochrome)
   - Perfect Stage 1 through 7 Visibility with Quick-Filter Stepper:
       * Stage 1 & 2: DeepSpill-Net (MobileNetV2-UNet) + 8-CCA Swath Filter
       * Stage 3: WGS84 Karney Geodetic Characterisation & PCA Ellipse
       * Stage 3b: Inverse Fay Viscous-Spreading Slick Age Estimation
       * Stage 5: Lagrangian Ocean Current Hindcast & Release Origin
       * Stage 5b: Forward 96h Dispersion Forecast & Envelopes
       * Stage 6: AIS Spatio-Temporal Candidate Track Gating
       * Stage 7: 5-Factor MCDA Suspect Vessel Attribution Ranking
   - Integrated Map API Token & Multi-Layer Leaflet Controller
   - High-Contrast Monochrome Tactical Map
   - Focused SAR Spill Imagery Inspection
   - Step-by-Step Slide Deck Presentation Mode
   - Live JSON File Artifacts Explorer & Downloader
=====================================================================================
"""

import os
import sys
import json
import socket
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Workspace Root
WORKSPACE_ROOT = Path(__file__).resolve().parent

# Ensure Windows command-line handles UTF-8 gracefully
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def find_free_port(start_port=5000, max_attempts=50):
    """Finds an available TCP port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("0.0.0.0", port)) != 0:
                return port
    return start_port


def discover_cases():
    """Discovers available spill cases in CASES/ and cases/."""
    cases = []
    seen = set()

    for folder_name in ["CASES", "cases"]:
        folder = WORKSPACE_ROOT / folder_name
        if folder.is_dir():
            for sub in sorted(folder.iterdir()):
                if sub.is_dir() and not sub.name.startswith("."):
                    case_id = sub.name
                    norm_id = case_id.lower().replace(" ", "_")
                    if norm_id not in seen:
                        seen.add(norm_id)
                        has_full = (sub / "case_full.json").exists()
                        has_spill = (sub / "SPILL.json").exists() or (sub / "spill.json").exists()
                        cases.append({
                            "id": case_id,
                            "folder": folder_name,
                            "path": str(sub.relative_to(WORKSPACE_ROOT)),
                            "has_full": has_full,
                            "has_spill": has_spill
                        })
    return cases


def load_case_data(case_id):
    """Loads complete case dataset, resolving case_full.json or individual stage JSONs."""
    target_dir = None
    for folder_name in ["CASES", "cases"]:
        p1 = WORKSPACE_ROOT / folder_name / case_id
        p2 = WORKSPACE_ROOT / folder_name / case_id.replace("_", " ")
        if p1.is_dir():
            target_dir = p1
            break
        elif p2.is_dir():
            target_dir = p2
            break

    if not target_dir or not target_dir.exists():
        return {"error": f"Case directory not found for: {case_id}"}

    # 1. Prefer case_full.json
    full_json_path = target_dir / "case_full.json"
    data = {}
    if full_json_path.exists():
        try:
            with open(full_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            data = {"error": f"Failed to parse case_full.json: {str(e)}"}

    if not data:
        data = {"case_id": case_id, "status": "partial"}

    # Supplement individual stage files if missing in data
    # Stage 3: Spill
    if "detection" not in data:
        for sname in ["SPILL.json", "spill.json"]:
            sp = target_dir / sname
            if sp.exists():
                try:
                    with open(sp, "r", encoding="utf-8") as f:
                        data["detection"] = json.load(f)
                        break
                except Exception:
                    pass

    # Stage 3b: Age
    if "age" not in data:
        for aname in ["age.json", "age_estimate.json"]:
            ap = target_dir / aname
            if ap.exists():
                try:
                    with open(ap, "r", encoding="utf-8") as f:
                        data["age"] = json.load(f)
                        break
                except Exception:
                    pass

    # Stage 5: Hindcast
    if "hindcast" not in data:
        hp = target_dir / "stage5" / "hindcast.json"
        if hp.exists():
            try:
                with open(hp, "r", encoding="utf-8") as f:
                    data["hindcast"] = json.load(f)
            except Exception:
                pass

    # Stage 5b: Forecast
    if "forecast" not in data:
        fp = target_dir / "forecast.json"
        if fp.exists():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data["forecast"] = json.load(f)
            except Exception:
                pass

    # Stage 5b: Forecast GeoJSON Envelope
    if "forecast_envelope" not in data:
        fep = target_dir / "forecast_envelope.geojson"
        if fep.exists():
            try:
                with open(fep, "r", encoding="utf-8") as f:
                    data["forecast_envelope"] = json.load(f)
            except Exception:
                pass

    # Stage 6: AIS Candidates
    if "ais_candidates" not in data and "ais_filtered" not in data:
        a6p = target_dir / "stage6" / "ais_filtered.json"
        if a6p.exists():
            try:
                with open(a6p, "r", encoding="utf-8") as f:
                    data["ais_filtered"] = json.load(f)
            except Exception:
                pass

    # Stage 7: Suspects
    if "suspects" not in data:
        sp7 = target_dir / "stage7" / "ranked_suspects.json"
        if sp7.exists():
            try:
                with open(sp7, "r", encoding="utf-8") as f:
                    data["suspects"] = json.load(f)
            except Exception:
                pass

    # Check available preview images
    images = []
    for img_file in target_dir.glob("*.png"):
        images.append(img_file.name)
    data["_available_images"] = images
    data["_case_path"] = str(target_dir.relative_to(WORKSPACE_ROOT))

    return data


def load_ml_metrics():
    """Loads precision, recall, dice score, accuracy from outputs/ and benchmarks."""
    metrics = {
        "deepspill_net": {
            "model_name": "DeepSpill-Net (MobileNetV2-UNet)",
            "architecture": "U-Net with Pre-trained MobileNetV2 Backbone",
            "parameters": "3.4M",
            "test_dice": 0.8828,
            "test_iou": 0.7926,
            "val_best_dice": 0.8855,
            "val_best_iou": 0.7971,
            "precision": 0.912,
            "recall": 0.858,
            "loss_function": "Hybrid Soft-Dice + BCEWithLogits",
            "tile_size": "256x256 (32px overlap)"
        },
        "sar_screennet": {
            "model_name": "SAR-ScreenNet (EfficientNet-B0)",
            "architecture": "Compound Scaled CNN Classifier",
            "parameters": "5.3M",
            "test_accuracy": 0.9585,
            "test_f1_score": 0.9387,
            "val_best_f1": 0.9390,
            "benchmark_dataset": "CSIRO Sentinel-1 Marine Oil Benchmark",
            "input_resolution": "224x224 Grayscale SAR"
        },
        "morphological_filter": {
            "name": "8-CCA Swath Artifact Filter",
            "algorithm": "Connected Components + Border Ratio Analysis",
            "min_pixel_area": 500,
            "swath_edge_rejection_pct": 30.0,
            "ocean_clutter_drop_pct": 5.0
        },
        "geodesic_pca": {
            "name": "Karney Ellipsoidal Geodesics + Bivariate Gaussian PCA",
            "ellipsoid": "WGS84",
            "orientation_frame": "Clockwise from True North"
        },
        "inverse_fay_age": {
            "model_name": "Inverse Fay Viscous-Spreading Model",
            "physics": "Gravity-Viscous spreading balance R(t) = k*(V*t)^(1/4)",
            "wind_coupling": "10-meter wind speed & temporal direction alignment",
            "operational_bounds_hours": [3.0, 72.0]
        },
        "lagrangian_dynamics": {
            "model_name": "OpenDrift / Lagrangian Particle Advection",
            "forward_particles": 5000,
            "windage_factor": 0.03,
            "turbulent_diffusion_Dh": "10.0 m^2/s",
            "metocean_sources": "INCOIS ROMS + Copernicus Global CMEMS"
        },
        "mcda_ranker": {
            "name": "5-Factor Multi-Criteria Decision Analysis",
            "weights": {
                "spatial_proximity_30pct": 0.30,
                "temporal_coincidence_25pct": 0.25,
                "drift_speed_match_20pct": 0.20,
                "dwell_loitering_15pct": 0.15,
                "vessel_risk_prior_10pct": 0.10
            }
        }
    }

    # Live overrides from outputs/
    unet_txt = WORKSPACE_ROOT / "outputs" / "unet_test_metrics.txt"
    if unet_txt.exists():
        try:
            with open(unet_txt, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        k, v = line.strip().split(":", 1)
                        try:
                            metrics["deepspill_net"][k.strip()] = float(v.strip())
                        except ValueError:
                            pass
        except Exception:
            pass

    csiro_txt = WORKSPACE_ROOT / "outputs" / "csiro_test_metrics.txt"
    if csiro_txt.exists():
        try:
            with open(csiro_txt, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        k, v = line.strip().split(":", 1)
                        try:
                            metrics["sar_screennet"][k.strip()] = float(v.strip())
                        except ValueError:
                            pass
        except Exception:
            pass

    return metrics


# ── ULTRA-ATTRACTIVE WHITE & BLACK INTERFACE ──────────────────────────────────
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="white">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SIH26143 — Arabian Sea Oil Spill Detection & Attribution System</title>

  <!-- Google Fonts & FontAwesome -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />

  <!-- Leaflet CSS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />

  <style>
    /* ── DUAL-THEME VARIABLES (WHITE & BLACK) ── */
    :root[data-theme="white"] {
      --bg-page: #f8fafc;
      --bg-header: #ffffff;
      --bg-card: #ffffff;
      --bg-card-subtle: #f1f5f9;
      --bg-banner: #f8fafc;
      --border-color: #e2e8f0;
      --border-strong: #0f172a;
      --text-main: #020617;
      --text-sub: #334155;
      --text-muted: #64748b;
      --accent-badge-bg: #020617;
      --accent-badge-text: #ffffff;
      --btn-primary-bg: #020617;
      --btn-primary-text: #ffffff;
      --btn-secondary-bg: #f1f5f9;
      --btn-secondary-text: #020617;
      --map-bg: #e2e8f0;
      --table-stripe: #f8fafc;
      --shadow-card: 0 4px 12px -2px rgba(0, 0, 0, 0.06), 0 2px 6px -1px rgba(0, 0, 0, 0.04);
      --shadow-header: 0 1px 3px 0 rgba(0, 0, 0, 0.08);
      --progress-track: #e2e8f0;
      --progress-fill: #020617;
    }

    :root[data-theme="black"] {
      --bg-page: #000000;
      --bg-header: #000000;
      --bg-card: #09090b;
      --bg-card-subtle: #18181b;
      --bg-banner: #09090b;
      --border-color: #27272a;
      --border-strong: #ffffff;
      --text-main: #ffffff;
      --text-sub: #e4e4e7;
      --text-muted: #a1a1aa;
      --accent-badge-bg: #ffffff;
      --accent-badge-text: #000000;
      --btn-primary-bg: #ffffff;
      --btn-primary-text: #000000;
      --btn-secondary-bg: #18181b;
      --btn-secondary-text: #ffffff;
      --map-bg: #000000;
      --table-stripe: #121215;
      --shadow-card: 0 4px 20px rgba(0, 0, 0, 0.8);
      --shadow-header: 0 1px 0 0 #27272a;
      --progress-track: #27272a;
      --progress-fill: #ffffff;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      scrollbar-width: thin;
      scrollbar-color: var(--border-color) transparent;
      transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
    }

    body {
      background-color: var(--bg-page);
      color: var(--text-main);
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      line-height: 1.5;
    }

    /* ── HEADER ── */
    header {
      background: var(--bg-header);
      border-bottom: 1px solid var(--border-color);
      box-shadow: var(--shadow-header);
      position: sticky;
      top: 0;
      z-index: 1000;
      padding: 0.9rem 1.75rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1.25rem;
    }

    .brand-group {
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }

    .brand-badge {
      background: var(--accent-badge-bg);
      color: var(--accent-badge-text);
      font-weight: 800;
      font-size: 0.75rem;
      padding: 0.35rem 0.65rem;
      border-radius: 6px;
      letter-spacing: 0.08em;
    }

    .brand-title {
      font-size: 1.15rem;
      font-weight: 800;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 0.5rem;
      letter-spacing: -0.02em;
    }

    .brand-subtitle {
      font-size: 0.76rem;
      color: var(--text-muted);
      font-weight: 500;
    }

    .header-nav {
      display: flex;
      align-items: center;
      gap: 0.65rem;
    }

    .case-selector {
      background: var(--bg-card);
      color: var(--text-main);
      border: 1px solid var(--border-color);
      padding: 0.5rem 1rem;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      outline: none;
    }

    .case-selector:hover, .case-selector:focus {
      border-color: var(--border-strong);
    }

    .nav-btn {
      background: var(--btn-secondary-bg);
      color: var(--btn-secondary-text);
      border: 1px solid var(--border-color);
      padding: 0.5rem 0.95rem;
      border-radius: 8px;
      font-size: 0.82rem;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }

    .nav-btn:hover {
      border-color: var(--border-strong);
    }

    .nav-btn.active {
      background: var(--btn-primary-bg);
      color: var(--btn-primary-text);
      border-color: var(--btn-primary-bg);
    }

    .theme-toggle-btn {
      background: var(--btn-secondary-bg);
      color: var(--btn-secondary-text);
      border: 1px solid var(--border-color);
      padding: 0.5rem 0.85rem;
      border-radius: 8px;
      font-size: 0.82rem;
      font-weight: 800;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }

    .theme-toggle-btn:hover {
      border-color: var(--border-strong);
    }

    /* ── MAIN LAYOUT ── */
    main {
      flex: 1;
      padding: 1.5rem 1.75rem;
      max-width: 1800px;
      margin: 0 auto;
      width: 100%;
    }

    /* ── TOP KPI TILES (WHITE & BLACK) ── */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    .kpi-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 1.25rem;
      box-shadow: var(--shadow-card);
      position: relative;
    }

    .kpi-card:hover {
      border-color: var(--border-strong);
    }

    .kpi-label {
      font-size: 0.72rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .kpi-val {
      font-size: 1.6rem;
      font-weight: 800;
      color: var(--text-main);
      margin: 0.4rem 0 0.25rem 0;
      font-family: 'JetBrains Mono', monospace;
      letter-spacing: -0.03em;
    }

    .kpi-sub {
      font-size: 0.78rem;
      color: var(--text-sub);
      font-weight: 600;
    }

    /* ── DASHBOARD GRID ── */
    .dashboard-layout {
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      gap: 1.5rem;
      align-items: start;
    }

    @media (max-width: 1100px) {
      .dashboard-layout {
        grid-template-columns: 1fr;
      }
    }

    /* Left: Map Card */
    .map-box {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      height: 800px;
      box-shadow: var(--shadow-card);
    }

    .card-header {
      padding: 1rem 1.35rem;
      background: var(--bg-card);
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
    }

    .card-header h3 {
      font-size: 0.95rem;
      font-weight: 800;
      display: flex;
      align-items: center;
      gap: 0.55rem;
      color: var(--text-main);
    }

    #map {
      flex: 1;
      width: 100%;
      background: var(--map-bg);
    }

    .map-legend {
      padding: 0.85rem 1.35rem;
      background: var(--bg-card);
      border-top: 1px solid var(--border-color);
      display: flex;
      flex-wrap: wrap;
      gap: 1.25rem;
      font-size: 0.78rem;
      color: var(--text-sub);
      font-weight: 600;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }

    .legend-color {
      width: 14px;
      height: 14px;
      border-radius: 3px;
    }

    /* Right: Stages Column */
    .stages-column {
      display: flex;
      flex-direction: column;
      gap: 1.1rem;
      max-height: 800px;
      overflow-y: auto;
      padding-right: 0.4rem;
    }

    /* Stepper Toolbar */
    .stage-filter-bar {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 0.45rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
      position: sticky;
      top: 0;
      z-index: 20;
      box-shadow: var(--shadow-card);
    }

    .stage-filter-btn {
      background: transparent;
      color: var(--text-muted);
      border: 1px solid transparent;
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 800;
      cursor: pointer;
      letter-spacing: 0.02em;
    }

    .stage-filter-btn:hover {
      color: var(--text-main);
      background: var(--bg-card-subtle);
    }

    .stage-filter-btn.active {
      background: var(--btn-primary-bg);
      color: var(--btn-primary-text);
    }

    /* Stage Cards */
    .stage-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: var(--shadow-card);
    }

    .stage-card:hover {
      border-color: var(--border-strong);
    }

    .stage-header {
      padding: 1rem 1.25rem;
      background: var(--bg-card);
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-color);
    }

    .stage-title-wrap {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .stage-num-badge {
      background: var(--accent-badge-bg);
      color: var(--accent-badge-text);
      font-weight: 800;
      font-size: 0.72rem;
      padding: 0.25rem 0.6rem;
      border-radius: 5px;
      letter-spacing: 0.06em;
    }

    .stage-title {
      font-size: 0.94rem;
      font-weight: 800;
      color: var(--text-main);
    }

    .status-pill {
      font-size: 0.7rem;
      font-weight: 800;
      padding: 0.25rem 0.65rem;
      border-radius: 999px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      background: var(--accent-badge-bg);
      color: var(--accent-badge-text);
    }

    .stage-body {
      padding: 1.25rem;
      background: var(--bg-card);
      font-size: 0.86rem;
      line-height: 1.6;
      color: var(--text-sub);
    }

    /* Formula & Algorithm Card */
    .math-banner {
      background: var(--bg-banner);
      border: 1px solid var(--border-color);
      border-left: 4px solid var(--text-main);
      border-radius: 8px;
      padding: 0.85rem 1.1rem;
      margin: 0.85rem 0;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }

    .math-eq {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.9rem;
      font-weight: 800;
      color: var(--text-main);
      letter-spacing: 0.01em;
    }

    .math-desc {
      font-size: 0.76rem;
      color: var(--text-muted);
      font-weight: 600;
    }

    /* Key-Value Tables */
    .metric-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      margin: 0.75rem 0;
    }

    .metric-table tr {
      border-bottom: 1px solid var(--border-color);
    }

    .metric-table tr:last-child {
      border-bottom: none;
    }

    .metric-table td {
      padding: 0.6rem 0.7rem;
      color: var(--text-sub);
      font-weight: 500;
    }

    .metric-table td.val {
      text-align: right;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 800;
      color: var(--text-main);
    }

    /* Image Preview Box */
    .image-viewer-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      overflow: hidden;
      margin-top: 1rem;
    }

    .image-viewer-toolbar {
      padding: 0.6rem 0.85rem;
      background: var(--bg-card-subtle);
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
    }

    .img-tab-btn {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      padding: 0.3rem 0.75rem;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 800;
      cursor: pointer;
    }

    .img-tab-btn.active {
      background: var(--btn-primary-bg);
      color: var(--btn-primary-text);
      border-color: var(--btn-primary-bg);
    }

    .preview-img-display {
      width: 100%;
      height: 230px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #000000;
      overflow: hidden;
      position: relative;
    }

    .preview-img-display img {
      max-width: 100%;
      max-height: 230px;
      object-fit: contain;
      display: block;
      cursor: zoom-in;
    }

    /* Suspect Vessel Cards */
    .suspect-item {
      background: var(--bg-card-subtle);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 0.95rem 1.1rem;
      margin-bottom: 0.75rem;
    }

    .suspect-item:hover {
      border-color: var(--border-strong);
    }

    .suspect-top-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.4rem;
    }

    .suspect-name {
      font-size: 0.94rem;
      font-weight: 800;
      color: var(--text-main);
    }

    .suspect-score-badge {
      background: var(--accent-badge-bg);
      color: var(--accent-badge-text);
      padding: 0.22rem 0.65rem;
      border-radius: 5px;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 800;
      font-size: 0.78rem;
    }

    .suspect-meta-row {
      font-size: 0.78rem;
      color: var(--text-muted);
      display: flex;
      flex-wrap: wrap;
      gap: 1.1rem;
      margin-bottom: 0.55rem;
      font-weight: 600;
    }

    .attr-bar-track {
      background: var(--progress-track);
      height: 7px;
      border-radius: 999px;
      overflow: hidden;
    }

    .attr-bar-fill {
      height: 100%;
      background: var(--progress-fill);
      border-radius: 999px;
    }

    /* ── SLIDE DECK (WHITE & BLACK) ── */
    .slides-container {
      display: none;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 2.5rem;
      box-shadow: var(--shadow-card);
      max-width: 1400px;
      margin: 0 auto;
    }

    .slide-controls {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 2rem;
      padding-bottom: 1.25rem;
      border-bottom: 1px solid var(--border-color);
    }

    .slide-step-indicator {
      display: flex;
      gap: 0.6rem;
    }

    .step-dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: var(--border-color);
      cursor: pointer;
    }

    .step-dot.active {
      background: var(--btn-primary-bg);
      width: 32px;
      border-radius: 6px;
    }

    .slide-page {
      display: none;
      animation: fadeIn 0.2s ease;
    }

    .slide-page.active {
      display: block;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .slide-title {
      font-size: 1.85rem;
      font-weight: 900;
      color: var(--text-main);
      margin-bottom: 0.4rem;
      letter-spacing: -0.03em;
    }

    .slide-subtitle {
      font-size: 0.98rem;
      color: var(--text-muted);
      font-weight: 700;
      margin-bottom: 1.75rem;
    }

    .slide-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.75rem;
    }

    @media (max-width: 900px) {
      .slide-grid {
        grid-template-columns: 1fr;
      }
    }

    .slide-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 1.5rem;
      box-shadow: var(--shadow-card);
    }

    .slide-card h4 {
      font-size: 1.05rem;
      font-weight: 800;
      margin-bottom: 1rem;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 0.55rem;
    }

    /* ── JSON INSPECTOR VIEW ── */
    .json-container {
      display: none;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 2rem;
      box-shadow: var(--shadow-card);
    }

    .json-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
      margin-bottom: 1.25rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border-color);
    }

    .json-tab-btn {
      background: var(--bg-card-subtle);
      color: var(--text-muted);
      border: 1px solid var(--border-color);
      padding: 0.45rem 0.9rem;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 700;
      cursor: pointer;
    }

    .json-tab-btn:hover {
      color: var(--text-main);
      border-color: var(--border-strong);
    }

    .json-tab-btn.active {
      background: var(--btn-primary-bg);
      color: var(--btn-primary-text);
      border-color: var(--btn-primary-bg);
    }

    pre.code-block {
      background: #000000;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 1.25rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.84rem;
      color: #ffffff;
      max-height: 640px;
      overflow: auto;
      white-space: pre-wrap;
    }

    /* Buttons */
    .btn {
      background: var(--btn-primary-bg);
      color: var(--btn-primary-text);
      font-weight: 800;
      border: 1px solid var(--btn-primary-bg);
      padding: 0.55rem 1.15rem;
      border-radius: 8px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      font-size: 0.84rem;
    }

    .btn:hover {
      opacity: 0.9;
    }

    .btn-secondary {
      background: var(--btn-secondary-bg);
      color: var(--btn-secondary-text);
      border: 1px solid var(--border-color);
    }

    .btn-secondary:hover {
      border-color: var(--border-strong);
    }

    .tag {
      display: inline-block;
      font-size: 0.72rem;
      font-weight: 800;
      padding: 0.22rem 0.55rem;
      border-radius: 5px;
      background: var(--bg-card-subtle);
      color: var(--text-main);
      border: 1px solid var(--border-color);
    }

    /* Leaflet Controls Styling */
    .leaflet-top {
      top: 14px !important;
    }

    .leaflet-control-layers {
      background: var(--bg-card) !important;
      border: 1px solid var(--border-color) !important;
      border-radius: 8px !important;
      color: var(--text-main) !important;
      box-shadow: var(--shadow-card) !important;
      font-family: inherit !important;
      font-size: 0.8rem !important;
      font-weight: 600 !important;
      padding: 0.75rem 0.95rem !important;
      max-height: 380px !important;
      overflow-y: auto !important;
      z-index: 800 !important;
    }

    .leaflet-control-layers-separator {
      border-top: 1px solid var(--border-color) !important;
      margin: 0.5rem 0 !important;
    }

    .leaflet-control-layers label {
      cursor: pointer;
      margin-bottom: 0.35rem;
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }

    .leaflet-control-layers input[type="radio"],
    .leaflet-control-layers input[type="checkbox"] {
      accent-color: var(--text-main);
      cursor: pointer;
    }

    /* ── FULLSCREEN MAP MODE ── */
    .map-box.fullscreen-mode {
      position: fixed !important;
      top: 0 !important;
      left: 0 !important;
      width: 100vw !important;
      height: 100vh !important;
      z-index: 99999 !important;
      border-radius: 0 !important;
      border: none !important;
      margin: 0 !important;
      box-shadow: none !important;
    }

    .map-box.fullscreen-mode #map {
      height: calc(100vh - 125px) !important;
    }

    body.has-fullscreen-map {
      overflow: hidden !important;
    }
  </style>
</head>
<body>

  <!-- ── HEADER ── -->
  <header>
    <div class="brand-group">
      <span class="brand-badge">SIH26143</span>
      <div>
        <div class="brand-title">
          <i class="fa-solid fa-satellite-dish"></i>
          Arabian Sea Spill Surveillance & Attribution
        </div>
        <div class="brand-subtitle">Sentinel-1 SAR Detection, Lagrangian Drift Simulation & AIS MCDA Engine</div>
      </div>
    </div>

    <div class="header-nav">
      <!-- Case Selector -->
      <select id="caseSelect" class="case-selector" onchange="switchCase(this.value)">
        <option value="">Loading cases...</option>
      </select>

      <!-- View Switchers -->
      <button class="nav-btn active" id="btnViewDashboard" onclick="switchView('dashboard')">
        <i class="fa-solid fa-chart-line"></i> Dashboard
      </button>
      <button class="nav-btn" id="btnViewSlides" onclick="switchView('slides')">
        <i class="fa-solid fa-person-chalkboard"></i> Slide Deck
      </button>
      <button class="nav-btn" id="btnViewJson" onclick="switchView('json')">
        <i class="fa-solid fa-code"></i> JSON Inspector
      </button>

      <!-- Theme Switcher (White Mode <-> Black Mode) -->
      <button class="theme-toggle-btn" id="themeToggleBtn" onclick="toggleTheme()">
        <i class="fa-solid fa-circle-half-stroke"></i> <span id="themeBtnText">Black Theme</span>
      </button>
    </div>
  </header>

  <!-- ── MAIN CONTENT ── -->
  <main>
    <!-- TOP KPI STRIP -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Detected Slick Area <i class="fa-solid fa-ruler-combined"></i></div>
        <div class="kpi-val" id="kpiArea">--</div>
        <div class="kpi-sub" id="kpiPixels">WGS84 Karney Geodesics</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Weathering Slick Age <i class="fa-solid fa-clock-rotate-left"></i></div>
        <div class="kpi-val" id="kpiAge">--</div>
        <div class="kpi-sub" id="kpiConfidence">Fay Inversion Model</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Top Suspect Vessel <i class="fa-solid fa-ship"></i></div>
        <div class="kpi-val" id="kpiSuspect" style="font-size: 1.25rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">--</div>
        <div class="kpi-sub" id="kpiSuspectScore">MCDA Attribution: --</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">96h Forward Drift <i class="fa-solid fa-compass"></i></div>
        <div class="kpi-val" id="kpiForecast">96 Hours</div>
        <div class="kpi-sub" id="kpiParticles">5,000 Lagrangian Particles</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">DeepSpill-Net Dice <i class="fa-solid fa-bullseye"></i></div>
        <div class="kpi-val" id="kpiDice">0.8828</div>
        <div class="kpi-sub">IoU: 0.7926 | Acc: 95.85%</div>
      </div>
    </div>

    <!-- ── VIEW 1: DASHBOARD (MAP + DETAILED STAGES) ── -->
    <div id="dashboardView" class="dashboard-layout">
      <!-- Left: Interactive Leaflet Tactical Map -->
      <div class="map-box">
        <div class="card-header">
          <h3>
            <i class="fa-solid fa-earth-asia"></i>
            Geospatial Marine Tactical Map
          </h3>
          <div style="display: flex; align-items: center; gap: 0.6rem;">
            <span class="tag">
              <i class="fa-solid fa-key"></i> Map API: Active
            </span>
            <button class="btn btn-secondary" style="padding: 0.35rem 0.8rem; font-size: 0.78rem;" id="btnFullscreenMap" onclick="toggleMapFullscreen()">
              <i class="fa-solid fa-expand"></i> <span id="fullscreenBtnText">Fullscreen</span>
            </button>
            <button class="btn btn-secondary" style="padding: 0.35rem 0.8rem; font-size: 0.78rem;" onclick="resetMapView()">
              <i class="fa-solid fa-crosshairs"></i> Recenter
            </button>
          </div>
        </div>

        <div id="map"></div>

        <div class="map-legend">
          <div class="legend-item">
            <div class="legend-color" style="background: #000000; border: 1px solid #fff;"></div>
            <span>Detected Slick Centroid</span>
          </div>
          <div class="legend-item">
            <div class="legend-color" style="background: #000000; border: 1px solid var(--text-main);"></div>
            <span>50% Core Forecast</span>
          </div>
          <div class="legend-item">
            <div class="legend-color" style="background: #a1a1aa; border: 1px dashed var(--text-main);"></div>
            <span>90% Outer Forecast</span>
          </div>
          <div class="legend-item">
            <div class="legend-color" style="background: transparent; border: 2px dashed var(--text-main);"></div>
            <span>Hindcast Origin Path</span>
          </div>
          <div class="legend-item">
            <div class="legend-color" style="background: #64748b;"></div>
            <span>AIS Candidate Vessels</span>
          </div>
        </div>
      </div>

      <!-- Right: Detailed Pipeline Stages 1 through 7 & 5b -->
      <div class="stages-column">
        <!-- Stage Stepper Toolbar -->
        <div class="stage-filter-bar">
          <button class="stage-filter-btn active" onclick="filterStageView('all')">ALL STAGES</button>
          <button class="stage-filter-btn" onclick="filterStageView('stage1_2')">1 & 2: SAR & FILTER</button>
          <button class="stage-filter-btn" onclick="filterStageView('stage3')">3: GEOMETRY</button>
          <button class="stage-filter-btn" onclick="filterStageView('stage3b')">3b: AGE PHYSICS</button>
          <button class="stage-filter-btn" onclick="filterStageView('stage5')">5: HINDCAST</button>
          <button class="stage-filter-btn" onclick="filterStageView('stage5b')">5b: 96h FORECAST</button>
          <button class="stage-filter-btn" onclick="filterStageView('stage6_7')">6 & 7: AIS & SUSPECTS</button>
        </div>

        <!-- Stage 1 & 2: Deep SAR Segmentation & Morphological Swath Filter -->
        <div class="stage-card" id="card-stage1_2">
          <div class="stage-header">
            <div class="stage-title-wrap">
              <span class="stage-num-badge">STAGE 1 & 2</span>
              <span class="stage-title">Deep SAR Segmentation & Swath Filter</span>
            </div>
            <span class="status-pill">PASS (COMPLETED)</span>
          </div>
          <div class="stage-body">
            <p><strong>DeepSpill-Net (MobileNetV2-UNet)</strong> evaluated on calibrated Sentinel-1 SAR intensity with <strong>256 &times; 256 px</strong> sliding window tiles (32 px overlap blend). Cleaned with 8-Connected Component Analysis and border drop-off filter.</p>

            <table class="metric-table">
              <tr><td>Test Dice Score</td><td class="val">0.8828 (88.3%)</td></tr>
              <tr><td>Test IoU Score</td><td class="val">0.7926 (79.3%)</td></tr>
              <tr><td>CSIRO ScreenNet Accuracy</td><td class="val">0.9585 (95.8%)</td></tr>
              <tr><td>CSIRO Oil F1-Score</td><td class="val">0.9387 (93.9%)</td></tr>
              <tr><td>Noise Cutoff Threshold</td><td class="val">&lt; 500 pixels dropped</td></tr>
              <tr><td>Swath Border Cutoff</td><td class="val">&gt; 30% border rejection</td></tr>
            </table>

            <div class="image-viewer-card">
              <div class="image-viewer-toolbar">
                <div style="font-size: 0.76rem; font-weight: 800; color: var(--text-main);">
                  <i class="fa-solid fa-image"></i> SAR Imagery Inspection
                </div>
                <div style="display: flex; gap: 0.45rem;">
                  <button class="img-tab-btn active" id="btnImgPreview" onclick="switchPreviewImage('preview')">
                    Spill Focus Preview
                  </button>
                  <button class="img-tab-btn" id="btnImgOverlay" onclick="switchPreviewImage('overlay')">
                    Full Swath Overlay
                  </button>
                </div>
              </div>
              <div class="preview-img-display">
                <img id="mainDisplayImg" src="" alt="SAR Spill Imagery" onclick="openImageModal()" />
              </div>
            </div>
          </div>
        </div>

        <!-- Stage 3: Spill Characterisation -->
        <div class="stage-card" id="card-stage3">
          <div class="stage-header">
            <div class="stage-title-wrap">
              <span class="stage-num-badge">STAGE 3</span>
              <span class="stage-title">Geodetic Spill Characterisation (WGS84 & PCA)</span>
            </div>
            <span class="status-pill">PASS</span>
          </div>
          <div class="stage-body">
            <p>Ellipsoidal geodesic contour integration and bivariate Gaussian PCA ellipse fitting for principal dispersion orientation.</p>
            <table class="metric-table">
              <tr><td>Centroid Coordinates</td><td class="val" id="stg3Centroid">--</td></tr>
              <tr><td>Surface Area</td><td class="val" id="stg3Area">--</td></tr>
              <tr><td>Perimeter</td><td class="val" id="stg3Perimeter">--</td></tr>
              <tr><td>Principal Axes</td><td class="val" id="stg3Axes">--</td></tr>
              <tr><td>Dispersion Orientation</td><td class="val" id="stg3Orient">--</td></tr>
              <tr><td>Bounding Box Lat/Lon</td><td class="val" id="stg3Bbox">--</td></tr>
            </table>
          </div>
        </div>

        <!-- Stage 3b: Oil Spill Age Estimation -->
        <div class="stage-card" id="card-stage3b">
          <div class="stage-header">
            <div class="stage-title-wrap">
              <span class="stage-num-badge">STAGE 3b</span>
              <span class="stage-title">Physical Age Estimation (Inverse Fay Spreading)</span>
            </div>
            <span class="status-pill">PASS</span>
          </div>
          <div class="stage-body">
            <div class="math-banner">
              <div class="math-eq">R(t) = k · (V · t)<sup>0.25</sup> &nbsp;⟹&nbsp; t<sub>age</sub> = (R / k)<sup>4</sup> / V</div>
              <div class="math-desc">Fay gravity-viscous expansion balance inverted for elapsed time using 10m wind speeds</div>
            </div>
            <table class="metric-table">
              <tr><td>Estimated Age Range</td><td class="val" id="stg3bRange">--</td></tr>
              <tr><td>Median Slick Age</td><td class="val" id="stg3bMedian">--</td></tr>
              <tr><td>Confidence Rating</td><td class="val" id="stg3bConf">--</td></tr>
              <tr><td>Compactness Index (4π·Area / Perimeter²)</td><td class="val" id="stg3bCompactness">--</td></tr>
              <tr><td>Operational Bounds</td><td class="val">[3.0h, 72.0h]</td></tr>
            </table>
          </div>
        </div>

        <!-- Stage 5: Ocean Current Hindcasting -->
        <div class="stage-card" id="card-stage5">
          <div class="stage-header">
            <div class="stage-title-wrap">
              <span class="stage-num-badge">STAGE 5</span>
              <span class="stage-title">Lagrangian Ocean Hindcast & Release Origin</span>
            </div>
            <span class="status-pill">PASS</span>
          </div>
          <div class="stage-body">
            <div class="math-banner">
              <div class="math-eq">v⃗<sub>total</sub> = u⃗<sub>ocean</sub>(t, x, y) + 0.03 · u⃗<sub>wind</sub>(t, x, y)</div>
              <div class="math-desc">Reverse-time advection of 30 Lagrangian particles coupled with 3% empirical windage factor</div>
            </div>
            <table class="metric-table">
              <tr><td>Estimated Release Origin</td><td class="val" id="stg5Origin">--</td></tr>
              <tr><td>Best Simulation Horizon</td><td class="val" id="stg5Duration">48 Hours</td></tr>
              <tr><td>Hindcast Confidence</td><td class="val" id="stg5Conf">58.98%</td></tr>
              <tr><td>Particle Spread Std Dev</td><td class="val" id="stg5Std">0.255 km</td></tr>
              <tr><td>Hydrodynamic Forcing</td><td class="val">Copernicus Global / INCOIS ROMS</td></tr>
            </table>
          </div>
        </div>

        <!-- Stage 5b: Forward Forecast (24h/72h/96h) -->
        <div class="stage-card" id="card-stage5b">
          <div class="stage-header">
            <div class="stage-title-wrap">
              <span class="stage-num-badge">STAGE 5b</span>
              <span class="stage-title">Forward Trajectory & Probability Envelopes (96h)</span>
            </div>
            <span class="status-pill">PASS</span>
          </div>
          <div class="stage-body">
            <div class="math-banner">
              <div class="math-eq">∂C/∂t = -∇·(u⃗C) + D<sub>h</sub>∇²C &nbsp;|&nbsp; D<sub>h</sub> = 10.0 m²/s</div>
              <div class="math-desc">Forward dispersion model seeding 5,000 particles across detected polygon with stochastic walk</div>
            </div>
            <table class="metric-table">
              <tr><td>Operational Horizons</td><td class="val">24h, 72h, 96h (INCOIS OOSA aligned)</td></tr>
              <tr><td>96h Forecast Centroid</td><td class="val" id="stg5bCentroid">--</td></tr>
              <tr><td>Diffusion Coefficient (Dh)</td><td class="val">10.0 m²/s</td></tr>
              <tr><td>Generated Probability Envelopes</td><td class="val">50% Core & 90% Outer</td></tr>
              <tr><td>Particles Seeded</td><td class="val">5,000 Georeferenced Particles</td></tr>
            </table>
          </div>
        </div>

        <!-- Stage 6 & 7: AIS Gating & Attribution Ranking -->
        <div class="stage-card" id="card-stage6_7">
          <div class="stage-header">
            <div class="stage-title-wrap">
              <span class="stage-num-badge">STAGE 6 & 7</span>
              <span class="stage-title">AIS Gating & 5-Factor MCDA Attribution</span>
            </div>
            <span class="status-pill">PASS</span>
          </div>
          <div class="stage-body">
            <div class="math-banner">
              <div class="math-eq">Score = 0.30·S<sub>prox</sub> + 0.25·S<sub>time</sub> + 0.20·S<sub>speed</sub> + 0.15·S<sub>dwell</sub> + 0.10·S<sub>type</sub></div>
              <div class="math-desc">Multi-Criteria Decision Analysis composite attribution index evaluating suspect vessels</div>
            </div>
            <table class="metric-table" style="margin-bottom: 0.75rem;">
              <tr><td>Gating Cone Buffer</td><td class="val">50.0 km radius from hindcast track</td></tr>
              <tr><td>Interpolation Method</td><td class="val">Spherical Great-Circle Geodesics</td></tr>
            </table>
            <div id="suspectsContainer">
              <!-- Populated via JS -->
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ── VIEW 2: PRESENTATION SLIDE DECK (WHITE & BLACK) ── -->
    <div id="slidesView" class="slides-container">
      <div class="slide-controls">
        <div style="display: flex; gap: 0.65rem;">
          <button class="btn btn-secondary" onclick="changeSlide(-1)">
            <i class="fa-solid fa-chevron-left"></i> Prev Slide
          </button>
          <button class="btn" onclick="changeSlide(1)">
            Next Slide <i class="fa-solid fa-chevron-right"></i>
          </button>
        </div>

        <div class="slide-step-indicator" id="slideDots">
          <div class="step-dot active" onclick="gotoSlide(0)"></div>
          <div class="step-dot" onclick="gotoSlide(1)"></div>
          <div class="step-dot" onclick="gotoSlide(2)"></div>
          <div class="step-dot" onclick="gotoSlide(3)"></div>
          <div class="step-dot" onclick="gotoSlide(4)"></div>
        </div>

        <div style="font-size: 0.86rem; font-weight: 800; color: var(--text-muted);" id="slideCounter">
          Slide 1 of 5
        </div>
      </div>

      <!-- Slide 1: Deep SAR AI (Stage 1 & 2) -->
      <div class="slide-page active" id="slide-0">
        <div class="slide-title">Stage 1 & 2: Deep SAR Segmentation & Morphological Swath Filter</div>
        <div class="slide-subtitle">DeepSpill-Net (MobileNetV2-UNet) + 8-CCA Swath Cleaning Pipeline</div>
        <div class="slide-grid">
          <div class="slide-card">
            <h4><i class="fa-solid fa-brain"></i> Neural Network Architecture</h4>
            <table class="metric-table">
              <tr><td>Encoder Backbone</td><td class="val">MobileNetV2 (Inverted Residuals)</td></tr>
              <tr><td>Decoder Type</td><td class="val">U-Net Skip Connections</td></tr>
              <tr><td>Model Parameters</td><td class="val">~3.4M params (Lightweight & Fast)</td></tr>
              <tr><td>Loss Function</td><td class="val">Soft Dice + BCEWithLogits Hybrid</td></tr>
              <tr><td>Inference Window</td><td class="val">256 &times; 256 px (32 px overlap blend)</td></tr>
            </table>
            <div style="margin-top: 1rem; display: flex; gap: 0.45rem; flex-wrap: wrap;">
              <span class="tag">SAR Single-Band dB Input</span>
              <span class="tag">Clean Ocean vs Slick</span>
              <span class="tag">Transfer Learning</span>
            </div>
          </div>
          <div class="slide-card">
            <h4><i class="fa-solid fa-chart-pie"></i> Validated Test Metrics</h4>
            <table class="metric-table">
              <tr><td>Test Dice Score</td><td class="val" style="font-size: 1.2rem;">0.8828 (88.3%)</td></tr>
              <tr><td>Test IoU Score</td><td class="val" style="font-size: 1.2rem;">0.7926 (79.3%)</td></tr>
              <tr><td>CSIRO ScreenNet Accuracy</td><td class="val" style="font-size: 1.2rem;">0.9585 (95.8%)</td></tr>
              <tr><td>CSIRO Oil F1-Score</td><td class="val" style="font-size: 1.2rem;">0.9387 (93.9%)</td></tr>
              <tr><td>Noise Cutoff Threshold</td><td class="val">&lt; 500 pixels dropped</td></tr>
            </table>
            <div style="margin-top: 0.85rem; border-radius: 8px; overflow: hidden; max-height: 180px; background: #000000; display: flex; justify-content: center; border: 1px solid var(--border-color);">
              <img id="slide1Img" src="" alt="SAR Detection Thumbnail" style="max-width: 100%; object-fit: contain;" />
            </div>
          </div>
        </div>
      </div>

      <!-- Slide 2: Geodetic Characterisation & Weathering (Stage 3 & 3b) -->
      <div class="slide-page" id="slide-1">
        <div class="slide-title">Stage 3 & 3b: Geodetic Characterisation & Weathering Age</div>
        <div class="slide-subtitle">Ellipsoidal PCA Moments & Inverse Fay Viscous-Spreading Physics</div>
        <div class="slide-grid">
          <div class="slide-card">
            <h4><i class="fa-solid fa-shapes"></i> Morphometric Measurements</h4>
            <p style="margin-bottom: 0.75rem; color: var(--text-muted); font-size: 0.85rem;">Calculated using Karney Geodesic Integrals over the WGS84 Reference Ellipsoid.</p>
            <table class="metric-table">
              <tr><td>Detected Area</td><td class="val" id="slide2Area">--</td></tr>
              <tr><td>Contour Perimeter</td><td class="val" id="slide2Perimeter">--</td></tr>
              <tr><td>Principal Axes</td><td class="val" id="slide2Axes">--</td></tr>
              <tr><td>Slick Orientation</td><td class="val" id="slide2Orient">--</td></tr>
              <tr><td>Centroid WGS84</td><td class="val" id="slide2Centroid">--</td></tr>
            </table>
          </div>
          <div class="slide-card">
            <h4><i class="fa-solid fa-hourglass-half"></i> Inverted Fay Spreading Theory</h4>
            <div class="math-banner">
              <div class="math-eq">R(t) = k · (V · t)<sup>0.25</sup> &nbsp;⟹&nbsp; t<sub>age</sub> = (R / k)<sup>4</sup> / V</div>
              <div class="math-desc">R = radius (km) | V = wind speed (km/h) | k = constant</div>
            </div>
            <table class="metric-table">
              <tr><td>Estimated Age Range</td><td class="val" id="slide2AgeRange">--</td></tr>
              <tr><td>Median Inferred Age</td><td class="val" id="slide2MedianAge">--</td></tr>
              <tr><td>Operational Bounds</td><td class="val">[3.0h, 72.0h]</td></tr>
              <tr><td>Alignment Verification</td><td class="val">Contiguous 10m Wind Direction Match</td></tr>
              <tr><td>Confidence Rating</td><td class="val" id="slide2Conf">--</td></tr>
            </table>
          </div>
        </div>
      </div>

      <!-- Slide 3: Ocean Dynamics (Stage 5 & 5b) -->
      <div class="slide-page" id="slide-2">
        <div class="slide-title">Stage 5 & 5b: Ocean Dynamic Hindcast & Forward Forecast</div>
        <div class="slide-subtitle">Lagrangian Backward Origin Backtrack + 96-Hour Dispersion Envelopes</div>
        <div class="slide-grid">
          <div class="slide-card">
            <h4><i class="fa-solid fa-backward"></i> Lagrangian Hindcast (Past Origin)</h4>
            <table class="metric-table">
              <tr><td>Simulation Horizons</td><td class="val">24h, 48h, 72h reverse trajectories</td></tr>
              <tr><td>Coupled Windage</td><td class="val">3.0% Empirical Surface Drift</td></tr>
              <tr><td>Hydrodynamic Data</td><td class="val">INCOIS ROMS / Copernicus Global</td></tr>
              <tr><td>Best Spill Origin</td><td class="val" id="slide3Origin">--</td></tr>
              <tr><td>Confidence Score</td><td class="val" id="slide3Conf">58.98%</td></tr>
            </table>
          </div>
          <div class="slide-card">
            <h4><i class="fa-solid fa-forward"></i> Forward 96h Dispersion (Future Forecast)</h4>
            <table class="metric-table">
              <tr><td>Particles Seeded</td><td class="val">5,000 across entire detected polygon</td></tr>
              <tr><td>Diffusion Coefficient</td><td class="val">Dh = 10.0 m²/s (Stochastic Walk)</td></tr>
              <tr><td>Forecast Horizons</td><td class="val">24h, 72h, 96h (INCOIS OOSA aligned)</td></tr>
              <tr><td>96h Centroid</td><td class="val" id="slide3Centroid">--</td></tr>
              <tr><td>Envelopes Output</td><td class="val">50% Core & 90% Outer</td></tr>
            </table>
          </div>
        </div>
      </div>

      <!-- Slide 4: AIS Gating & MCDA Ranking (Stage 6 & 7) -->
      <div class="slide-page" id="slide-3">
        <div class="slide-title">Stage 6 & 7: AIS Gating & Suspect Vessel Attribution</div>
        <div class="slide-subtitle">Spatio-Temporal Candidate Filtering & 5-Factor MCDA Ranking</div>
        <div class="slide-grid">
          <div class="slide-card">
            <h4><i class="fa-solid fa-filter"></i> Spatio-Temporal AIS Gating</h4>
            <table class="metric-table">
              <tr><td>Gating Cone Buffer</td><td class="val">50.0 km radius from hindcast track</td></tr>
              <tr><td>Interpolation Method</td><td class="val">Spherical Great-Circle Geodesics</td></tr>
              <tr><td>Temporal Window</td><td class="val">Coincident with inferred release epoch</td></tr>
              <tr><td>Candidate Vessels</td><td class="val" id="slide4Count">3 Filtered Vessels</td></tr>
            </table>
          </div>
          <div class="slide-card">
            <h4><i class="fa-solid fa-scale-balanced"></i> 5-Factor MCDA Formulation</h4>
            <div class="math-banner">
              <div class="math-eq">Score = 0.30·Prox + 0.25·Time + 0.20·Speed + 0.15·Dwell + 0.10·Type</div>
            </div>
            <div id="slide4Suspects">
              <!-- Populated via JS -->
            </div>
          </div>
        </div>
      </div>

      <!-- Slide 5: Complete Incident Executive Summary -->
      <div class="slide-page" id="slide-4">
        <div class="slide-title">Stage 1 - 7: Complete Incident Attribution Dossier</div>
        <div class="slide-subtitle">Operational Executive Summary & Maritime Authority Evidence Package</div>
        <div class="slide-card">
          <h4><i class="fa-solid fa-clipboard-check"></i> End-to-End Pipeline Audit</h4>
          <table class="metric-table">
            <thead>
              <tr style="border-bottom: 2px solid var(--border-color);">
                <th style="padding: 0.65rem; text-align: left; color: var(--text-muted);">Stage</th>
                <th style="padding: 0.65rem; text-align: left; color: var(--text-muted);">Description</th>
                <th style="padding: 0.65rem; text-align: left; color: var(--text-muted);">Algorithm / Model</th>
                <th style="padding: 0.65rem; text-align: right; color: var(--text-muted);">Status</th>
              </tr>
            </thead>
            <tbody>
              <tr><td style="font-weight: 800; color: var(--text-main);">Stage 1</td><td>SAR Semantic Segmentation</td><td>DeepSpill-Net (MobileNetV2-UNet)</td><td class="val" style="color: var(--text-main);">PASS (COMPLETED)</td></tr>
              <tr><td style="font-weight: 800; color: var(--text-main);">Stage 2</td><td>Swath Artifact Filter</td><td>8-CCA Morphological Cut</td><td class="val" style="color: var(--text-main);">PASS (COMPLETED)</td></tr>
              <tr><td style="font-weight: 800; color: var(--text-main);">Stage 3</td><td>Spill Characterisation</td><td>Karney Geodesics & PCA Ellipse</td><td class="val" style="color: var(--text-main);">PASS</td></tr>
              <tr><td style="font-weight: 800; color: var(--text-main);">Stage 3b</td><td>Weathering Age Estimation</td><td>Inverse Fay Spreading Model</td><td class="val" style="color: var(--text-main);">PASS</td></tr>
              <tr><td style="font-weight: 800; color: var(--text-main);">Stage 5</td><td>Ocean Current Hindcast</td><td>Lagrangian Particle Advection</td><td class="val" style="color: var(--text-main);">PASS</td></tr>
              <tr><td style="font-weight: 800; color: var(--text-main);">Stage 5b</td><td>Forward Forecast (96h)</td><td>Stochastic Dispersion Envelopes</td><td class="val" style="color: var(--text-main);">PASS</td></tr>
              <tr><td style="font-weight: 800; color: var(--text-main);">Stage 6</td><td>AIS Candidate Gating</td><td>Spatio-Temporal Distance Cone</td><td class="val" style="color: var(--text-main);">PASS</td></tr>
              <tr><td style="font-weight: 800; color: var(--text-main);">Stage 7</td><td>Suspect Attribution</td><td>5-Factor MCDA Ranker</td><td class="val" style="color: var(--text-main);">PASS</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ── VIEW 3: RAW JSON & ARTIFACTS INSPECTOR ── -->
    <div id="jsonView" class="json-container">
      <div class="card-header" style="margin: -2rem -2rem 1.25rem -2rem; border-radius: 12px 12px 0 0;">
        <h3><i class="fa-solid fa-file-code"></i> Pipeline Stage JSON Artifacts Explorer</h3>
        <div style="display: flex; gap: 0.5rem;">
          <button class="btn btn-secondary" onclick="copyJsonToClipboard()">
            <i class="fa-solid fa-copy"></i> Copy JSON
          </button>
          <button class="btn" onclick="downloadCurrentJson()">
            <i class="fa-solid fa-download"></i> Download File
          </button>
        </div>
      </div>

      <div class="json-tabs" id="jsonTabsContainer">
        <button class="json-tab-btn active" onclick="loadJsonTab('case_full.json')">case_full.json (Master)</button>
        <button class="json-tab-btn" onclick="loadJsonTab('SPILL.json')">SPILL.json (Stage 3)</button>
        <button class="json-tab-btn" onclick="loadJsonTab('age.json')">age.json (Stage 3b)</button>
        <button class="json-tab-btn" onclick="loadJsonTab('stage5/hindcast.json')">hindcast.json (Stage 5)</button>
        <button class="json-tab-btn" onclick="loadJsonTab('forecast.json')">forecast.json (Stage 5b)</button>
        <button class="json-tab-btn" onclick="loadJsonTab('forecast_envelope.geojson')">forecast_envelope.geojson</button>
        <button class="json-tab-btn" onclick="loadJsonTab('stage6/ais_filtered.json')">ais_filtered.json (Stage 6)</button>
        <button class="json-tab-btn" onclick="loadJsonTab('stage7/ranked_suspects.json')">ranked_suspects.json (Stage 7)</button>
      </div>

      <pre class="code-block" id="jsonCodeViewer">Loading JSON payload...</pre>
    </div>
  </main>

  <!-- Leaflet JS -->
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <script>
    // State
    let currentTheme = "white"; // "white" or "black"
    let currentCaseId = "";
    let currentCaseData = null;
    let map = null;
    let mapLayers = {};
    let currentSlideIdx = 0;
    const totalSlides = 5;
    let activeJsonFile = "case_full.json";
    let activeImageType = "preview"; // 'preview' or 'overlay'

    // Integrated Map API Token
    const MAP_API_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhIjoiYWNfYmY2eXczOG8iLCJqdGkiOiI2ZTA3ZThhYSJ9.xrEfRIbxz_lbtQ6dc2URWX5qnfvz9aDSD3YZSshHUxc";

    window.addEventListener("DOMContentLoaded", () => {
      initMap();
      fetchCases();
      setupKeyboardNav();
    });

    // ── THEME TOGGLE (WHITE & BLACK) ──
    function toggleTheme() {
      currentTheme = (currentTheme === "white") ? "black" : "white";
      document.documentElement.setAttribute("data-theme", currentTheme);
      document.getElementById("themeBtnText").textContent = (currentTheme === "white") ? "Black Theme" : "White Theme";

      // Refresh map layers to match theme
      if (currentCaseData) {
        renderMapLayers(currentCaseData);
      }
    }

    // ── LEAFLET MAP INITIALIZATION & LAYER CONTROLLER ──
    function initMap() {
      map = L.map("map", {
        zoomControl: true,
        attributionControl: true
      }).setView([10.55, 76.05], 9);

      // 1. High-Contrast Dark Matter Base Layer
      const darkTactical = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        maxZoom: 19,
        subdomains: "abcd",
        attribution: "&copy; CartoDB &copy; OpenStreetMap contributors"
      });

      // 2. High-Definition Satellite Imagery (ESRI)
      const satelliteHD = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
        maxZoom: 19,
        attribution: "&copy; Esri World Imagery"
      });

      // 3. Custom Authenticated Map Tile API (Powered by provided token)
      const customApiLayer = L.tileLayer(`https://tile.jawg.io/jawg-dark/{z}/{x}/{y}.png?access-token=${MAP_API_TOKEN}`, {
        maxZoom: 19,
        attribution: "&copy; Authenticated Map API"
      });

      // 4. Clean Positron / Monochrome White Layer
      const lightPositron = L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
        maxZoom: 19,
        subdomains: "abcd",
        attribution: "&copy; CartoDB Positron"
      });

      // Default Active Base Layer
      darkTactical.addTo(map);

      // Marine Seamarks Overlay
      const seamarksOverlay = L.tileLayer("https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png", {
        maxZoom: 18,
        attribution: "&copy; OpenSeaMap Navigation Marks"
      });

      // Initialize Layer Groups
      mapLayers.spillGroup = L.layerGroup().addTo(map);
      mapLayers.hindcastGroup = L.layerGroup().addTo(map);
      mapLayers.forecastGroup = L.layerGroup().addTo(map);
      mapLayers.vesselGroup = L.layerGroup().addTo(map);
      mapLayers.seamarksGroup = seamarksOverlay;

      // Base Maps Configuration
      const baseMaps = {
        "🌊 Dark Matter Tactical": darkTactical,
        "⚪ Clean Monochrome White": lightPositron,
        "🛰️ Satellite HD (ESRI)": satelliteHD,
        "🌐 Custom Map API": customApiLayer
      };

      // Overlays Configuration
      const overlayMaps = {
        "⚫ Detected Slick & Bbox": mapLayers.spillGroup,
        "🚨 50% & 90% Forecast Envelopes": mapLayers.forecastGroup,
        "⚪ Hindcast Origin Path": mapLayers.hindcastGroup,
        "🚢 AIS Suspect Vessels": mapLayers.vesselGroup,
        "⚓ Nautical Seamarks": mapLayers.seamarksGroup
      };

      // Add Layer Controller (collapsed: true prevents overflowing upwards)
      L.control.layers(baseMaps, overlayMaps, {
        position: "topright",
        collapsed: true
      }).addTo(map);
    }

    function resetMapView() {
      if (currentCaseData && currentCaseData.detection) {
        const d = currentCaseData.detection;
        map.setView([d.lat || d.centroid_lat, d.lon || d.centroid_lon], 9);
      }
    }

    // ── FULLSCREEN MAP TOGGLE ──
    let isMapFullscreen = false;

    function toggleMapFullscreen() {
      const mapBox = document.querySelector(".map-box");
      const btnText = document.getElementById("fullscreenBtnText");
      const btnIcon = document.querySelector("#btnFullscreenMap i");

      isMapFullscreen = !isMapFullscreen;
      mapBox.classList.toggle("fullscreen-mode", isMapFullscreen);
      document.body.classList.toggle("has-fullscreen-map", isMapFullscreen);

      if (isMapFullscreen) {
        btnText.textContent = "Exit Fullscreen";
        btnIcon.className = "fa-solid fa-compress";
      } else {
        btnText.textContent = "Fullscreen";
        btnIcon.className = "fa-solid fa-expand";
      }

      setTimeout(() => {
        map.invalidateSize();
      }, 150);
    }

    // Escape key exits fullscreen
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && isMapFullscreen) {
        toggleMapFullscreen();
      }
    });

    // ── STAGE FILTER CONTROLS ──
    function filterStageView(stageKey) {
      const buttons = document.querySelectorAll(".stage-filter-btn");
      buttons.forEach(btn => {
        btn.classList.toggle("active", btn.getAttribute("onclick").includes(`'${stageKey}'`));
      });

      const cardIds = [
        "card-stage1_2",
        "card-stage3",
        "card-stage3b",
        "card-stage5",
        "card-stage5b",
        "card-stage6_7"
      ];

      cardIds.forEach(id => {
        const card = document.getElementById(id);
        if (!card) return;
        if (stageKey === "all") {
          card.style.display = "block";
        } else {
          card.style.display = id === `card-${stageKey}` ? "block" : "none";
        }
      });
    }

    // ── CASES MANAGEMENT ──
    function fetchCases() {
      fetch("/api/cases")
        .then(res => res.json())
        .then(cases => {
          const sel = document.getElementById("caseSelect");
          sel.innerHTML = "";
          if (!cases || cases.length === 0) {
            sel.innerHTML = "<option>No cases found</option>";
            return;
          }

          cases.forEach(c => {
            const opt = document.createElement("option");
            opt.value = c.id;
            opt.textContent = `${c.id} (${c.folder})`;
            sel.appendChild(opt);
          });

          currentCaseId = cases[0].id;
          sel.value = currentCaseId;
          loadCase(currentCaseId);
        })
        .catch(err => {
          console.error("Error fetching cases:", err);
          document.getElementById("caseSelect").innerHTML = "<option>Error loading cases</option>";
        });
    }

    function switchCase(caseId) {
      if (!caseId) return;
      currentCaseId = caseId;
      loadCase(caseId);
    }

    function loadCase(caseId) {
      fetch(`/api/case?id=${encodeURIComponent(caseId)}`)
        .then(res => res.json())
        .then(data => {
          currentCaseData = data;
          renderCaseData(data);
          renderMapLayers(data);
          updatePreviewImage();
          loadJsonTab(activeJsonFile);
        })
        .catch(err => {
          console.error("Failed to load case data:", err);
        });
    }

    // ── RENDER DOM DATA ──
    function renderCaseData(d) {
      // 1. Detection
      if (d.detection) {
        const area = d.detection.area_km2 !== undefined ? d.detection.area_km2.toFixed(2) : "--";
        const px = d.detection.area_pixels !== undefined ? d.detection.area_pixels.toLocaleString() + " px" : "";
        document.getElementById("kpiArea").textContent = `${area} km²`;
        document.getElementById("kpiPixels").textContent = px;

        const lat = (d.detection.lat || d.detection.centroid_lat || 0).toFixed(4);
        const lon = (d.detection.lon || d.detection.centroid_lon || 0).toFixed(4);
        document.getElementById("stg3Centroid").textContent = `(${lat}° N, ${lon}° E)`;
        document.getElementById("stg3Area").textContent = `${area} km² (${px})`;
        document.getElementById("stg3Perimeter").textContent = `${(d.detection.perimeter_km || 0).toFixed(2)} km`;
        document.getElementById("stg3Axes").textContent = `a = ${(d.detection.major_axis_km || 0).toFixed(2)} km | b = ${(d.detection.minor_axis_km || 0).toFixed(2)} km`;
        document.getElementById("stg3Orient").textContent = `${(d.detection.orientation_deg || 0).toFixed(1)}° from True North`;

        if (d.detection.bbox_latlon) {
          const [w, s, e, n] = d.detection.bbox_latlon;
          document.getElementById("stg3Bbox").textContent = `[${w.toFixed(2)}°, ${s.toFixed(2)}°] to [${e.toFixed(2)}°, ${n.toFixed(2)}°]`;
        } else {
          document.getElementById("stg3Bbox").textContent = "--";
        }

        // Slide 2
        document.getElementById("slide2Area").textContent = `${area} km²`;
        document.getElementById("slide2Perimeter").textContent = `${(d.detection.perimeter_km || 0).toFixed(2)} km`;
        document.getElementById("slide2Axes").textContent = `a: ${(d.detection.major_axis_km || 0).toFixed(2)} km, b: ${(d.detection.minor_axis_km || 0).toFixed(2)} km`;
        document.getElementById("slide2Orient").textContent = `${(d.detection.orientation_deg || 0).toFixed(1)}° (PCA)`;
        document.getElementById("slide2Centroid").textContent = `(${lat}°, ${lon}°)`;
      }

      // 2. Age
      if (d.age) {
        const range = d.age.age_hours_range ? `${d.age.age_hours_range[0]} - ${d.age.age_hours_range[1]}h` : "--";
        const med = d.age.median_age_hours !== undefined ? `${d.age.median_age_hours}h` : "--";
        const conf = (d.age.confidence || "--").toUpperCase();
        document.getElementById("kpiAge").textContent = range;
        document.getElementById("kpiConfidence").textContent = `Conf: ${conf} | Median: ${med}`;
        document.getElementById("stg3bRange").textContent = range;
        document.getElementById("stg3bMedian").textContent = med;
        document.getElementById("stg3bConf").textContent = conf;
        document.getElementById("stg3bCompactness").textContent = (d.age.compactness || 0).toFixed(4);

        // Slide 2 Age
        document.getElementById("slide2AgeRange").textContent = range;
        document.getElementById("slide2MedianAge").textContent = med;
        document.getElementById("slide2Conf").textContent = conf;
      }

      // 3. Hindcast
      if (d.hindcast && d.hindcast.simulations && d.hindcast.simulations.length > 0) {
        const sim = d.hindcast.simulations[1] || d.hindcast.simulations[0];
        const origLat = sim.origin.lat.toFixed(4);
        const origLon = sim.origin.lon.toFixed(4);
        document.getElementById("stg5Origin").textContent = `(${origLat}° N, ${origLon}° E)`;
        document.getElementById("stg5Duration").textContent = `${sim.duration_hours} Hours`;
        document.getElementById("stg5Conf").textContent = `${((sim.confidence || 0.5898) * 100).toFixed(1)}%`;
        document.getElementById("stg5Std").textContent = `${(sim.particle_std_km || 0.25).toFixed(3)} km`;

        // Slide 3
        document.getElementById("slide3Origin").textContent = `(${origLat}° N, ${origLon}° E)`;
        document.getElementById("slide3Conf").textContent = `${((sim.confidence || 0.5898) * 100).toFixed(1)}%`;
      }

      // 4. Forecast 96h
      if (d.forecast && d.forecast.horizons && d.forecast.horizons["96h"]) {
        const h96 = d.forecast.horizons["96h"];
        const c96Lat = h96.centroid.lat.toFixed(4);
        const c96Lon = h96.centroid.lon.toFixed(4);
        document.getElementById("stg5bCentroid").textContent = `(${c96Lat}° N, ${c96Lon}° E)`;
        document.getElementById("slide3Centroid").textContent = `(${c96Lat}° N, ${c96Lon}° E)`;
      }

      // 5. Suspects
      let suspectsList = [];
      if (Array.isArray(d.suspects)) {
        suspectsList = d.suspects;
      } else if (d.suspects && Array.isArray(d.suspects.suspects)) {
        suspectsList = d.suspects.suspects;
      }

      if (suspectsList.length > 0) {
        const top = suspectsList[0];
        const topName = top.vessel_name || top.name || "Unknown Vessel";
        const topScore = top.attribution_score !== undefined ? (top.attribution_score * 100).toFixed(1) : "--";
        document.getElementById("kpiSuspect").textContent = topName;
        document.getElementById("kpiSuspectScore").textContent = `MCDA Score: ${topScore}%`;

        let html = "";
        suspectsList.slice(0, 3).forEach((s, idx) => {
          const name = s.vessel_name || s.name || `Vessel #${idx+1}`;
          const score = s.attribution_score !== undefined ? (s.attribution_score * 100).toFixed(1) : "--";
          const type = s.vessel_type || "Commercial Vessel";
          const dist = s.min_distance_km !== undefined ? s.min_distance_km.toFixed(1) : (s.distance_km || "--");
          html += `
            <div class="suspect-item">
              <div class="suspect-top-row">
                <span class="suspect-name">#${idx+1} ${name}</span>
                <span class="suspect-score-badge">${score}% Attribution</span>
              </div>
              <div class="suspect-meta-row">
                <span><i class="fa-solid fa-ship"></i> ${type}</span>
                <span><i class="fa-solid fa-location-arrow"></i> ${dist} km to origin</span>
                <span><i class="fa-solid fa-id-card"></i> MMSI: ${s.mmsi || '--'}</span>
              </div>
              <div class="attr-bar-track">
                <div class="attr-bar-fill" style="width: ${score}%;"></div>
              </div>
            </div>
          `;
        });
        document.getElementById("suspectsContainer").innerHTML = html;
        document.getElementById("slide4Suspects").innerHTML = html;
        document.getElementById("slide4Count").textContent = `${suspectsList.length} Filtered Vessels`;
      } else {
        document.getElementById("kpiSuspect").textContent = "None Filtered";
        document.getElementById("kpiSuspectScore").textContent = "MCDA: --";
        document.getElementById("suspectsContainer").innerHTML = "<p style='color: var(--text-muted);'>No AIS candidate tracks within spat-temporal cone.</p>";
      }
    }

    // ── IMAGE SWITCHER & RENDERING ──
    function switchPreviewImage(type) {
      activeImageType = type;
      document.getElementById("btnImgPreview").classList.toggle("active", type === "preview");
      document.getElementById("btnImgOverlay").classList.toggle("active", type === "overlay");
      updatePreviewImage();
    }

    function updatePreviewImage() {
      const img = document.getElementById("mainDisplayImg");
      const slide1Img = document.getElementById("slide1Img");
      const src = `/api/image?case=${encodeURIComponent(currentCaseId)}&name=${activeImageType}`;
      img.src = src;
      img.onerror = () => {
        if (activeImageType === "overlay") {
          img.src = `/api/image?case=${encodeURIComponent(currentCaseId)}&name=preview`;
        }
      };
      if (slide1Img) {
        slide1Img.src = `/api/image?case=${encodeURIComponent(currentCaseId)}&name=preview`;
      }
    }

    function openImageModal() {
      const src = `/api/image?case=${encodeURIComponent(currentCaseId)}&name=${activeImageType}`;
      window.open(src, "_blank");
    }

    // ── MAP LAYERS ──
    function renderMapLayers(d) {
      mapLayers.spillGroup.clearLayers();
      mapLayers.hindcastGroup.clearLayers();
      mapLayers.forecastGroup.clearLayers();
      mapLayers.vesselGroup.clearLayers();

      if (!d || !d.detection) return;

      const lat = d.detection.lat || d.detection.centroid_lat;
      const lon = d.detection.lon || d.detection.centroid_lon;
      if (!lat || !lon) return;

      const isLight = (currentTheme === "white");
      const strokeColor = isLight ? "#000000" : "#ffffff";
      const fillColor = isLight ? "#000000" : "#ffffff";

      // 1. Slick Centroid Marker
      const spillMarker = L.circleMarker([lat, lon], {
        radius: 9,
        color: "#ffffff",
        weight: 2.5,
        fillColor: "#000000",
        fillOpacity: 1.0
      }).bindPopup(`
        <strong style="color: #000;">Oil Spill Centroid</strong><br>
        <strong>Case:</strong> ${d.case_id || currentCaseId}<br>
        <strong>Area:</strong> ${(d.detection.area_km2 || 0).toFixed(2)} km²<br>
        <strong>Coords:</strong> ${lat.toFixed(4)}° N, ${lon.toFixed(4)}° E
      `);
      mapLayers.spillGroup.addLayer(spillMarker);

      // Bounding Box
      if (d.detection.bbox_latlon) {
        const [minLon, minLat, maxLon, maxLat] = d.detection.bbox_latlon;
        const bboxRect = L.rectangle([[minLat, minLon], [maxLat, maxLon]], {
          color: strokeColor,
          weight: 1.5,
          fillColor: fillColor,
          fillOpacity: 0.08,
          dashArray: "4, 4"
        });
        mapLayers.spillGroup.addLayer(bboxRect);
      }

      // 2. Forecast Envelopes
      if (d.forecast_envelope && d.forecast_envelope.features) {
        const geoLayer = L.geoJSON(d.forecast_envelope, {
          style: (feature) => {
            const is50 = (feature.properties.probability === "50%");
            return {
              color: is50 ? strokeColor : "#71717a",
              fillColor: is50 ? fillColor : "#a1a1aa",
              fillOpacity: is50 ? 0.35 : 0.15,
              weight: is50 ? 2.5 : 1.5,
              dashArray: is50 ? null : "4, 4"
            };
          },
          onEachFeature: (feature, layer) => {
            const p = feature.properties || {};
            layer.bindPopup(`
              <strong style="color: #000;">Forecast Dispersion Envelope</strong><br>
              <strong>Horizon:</strong> ${p.horizon || '--'}<br>
              <strong>Probability:</strong> ${p.probability || '--'}<br>
              <strong>Forecast Time:</strong> ${p.forecast_time || '--'}
            `);
          }
        });
        mapLayers.forecastGroup.addLayer(geoLayer);
      }

      // 3. Hindcast Origin & Trajectory Line
      if (d.hindcast && d.hindcast.simulations) {
        d.hindcast.simulations.forEach(sim => {
          if (sim.trajectory && sim.trajectory.length > 0) {
            const pts = sim.trajectory.map(pt => [pt.lat, pt.lon]);
            const poly = L.polyline(pts, {
              color: strokeColor,
              weight: 2.5,
              dashArray: "6, 6"
            });
            mapLayers.hindcastGroup.addLayer(poly);
          }

          if (sim.origin) {
            const originMarker = L.circleMarker([sim.origin.lat, sim.origin.lon], {
              radius: 7,
              color: "#ffffff",
              weight: 2,
              fillColor: "#000000",
              fillOpacity: 0.9
            }).bindPopup(`
              <strong style="color: #000;">Estimated Release Origin</strong><br>
              <strong>Duration:</strong> ${sim.duration_hours}h Hindcast<br>
              <strong>Coords:</strong> ${sim.origin.lat.toFixed(4)}° N, ${sim.origin.lon.toFixed(4)}° E<br>
              <strong>Confidence:</strong> ${((sim.confidence || 0.5898) * 100).toFixed(1)}%
            `);
            mapLayers.hindcastGroup.addLayer(originMarker);
          }
        });
      }

      // 4. Forward Forecast Line to 96h Centroid
      if (d.forecast && d.forecast.horizons && d.forecast.horizons["96h"]) {
        const c96 = d.forecast.horizons["96h"].centroid;
        const forecastLine = L.polyline([[lat, lon], [c96.lat, c96.lon]], {
          color: strokeColor,
          weight: 3
        });
        mapLayers.forecastGroup.addLayer(forecastLine);

        const m96 = L.circleMarker([c96.lat, c96.lon], {
          radius: 6,
          color: "#ffffff",
          weight: 2,
          fillColor: "#334155",
          fillOpacity: 0.9
        }).bindPopup(`
          <strong style="color: #000;">96h Forecast Centroid</strong><br>
          <strong>Coords:</strong> ${c96.lat.toFixed(4)}° N, ${c96.lon.toFixed(4)}° E
        `);
        mapLayers.forecastGroup.addLayer(m96);
      }

      // 5. Suspect Vessels Markers
      let suspectsList = [];
      if (Array.isArray(d.suspects)) suspectsList = d.suspects;
      else if (d.suspects && Array.isArray(d.suspects.suspects)) suspectsList = d.suspects.suspects;

      suspectsList.forEach(s => {
        const sLat = s.lat || s.last_lat || (s.coordinates ? s.coordinates[1] : null);
        const sLon = s.lon || s.last_lon || (s.coordinates ? s.coordinates[0] : null);
        if (sLat && sLon) {
          const shipIcon = L.divIcon({
            html: `<i class="fa-solid fa-ship" style="color: ${strokeColor}; font-size: 16px; text-shadow: 0 0 5px rgba(0,0,0,0.5);"></i>`,
            className: "ship-icon",
            iconSize: [20, 20],
            iconAnchor: [10, 10]
          });
          const m = L.marker([sLat, sLon], { icon: shipIcon }).bindPopup(`
            <strong style="color: #000;">${s.vessel_name || s.name || 'Candidate Vessel'}</strong><br>
            <strong>MMSI:</strong> ${s.mmsi || '--'}<br>
            <strong>Type:</strong> ${s.vessel_type || 'Cargo'}<br>
            <strong>Attribution Score:</strong> ${(s.attribution_score * 100).toFixed(1)}%
          `);
          mapLayers.vesselGroup.addLayer(m);
        }
      });

      map.setView([lat, lon], 9);
    }

    // ── VIEW SWITCHER ──
    function switchView(viewName) {
      document.getElementById("btnViewDashboard").classList.toggle("active", viewName === "dashboard");
      document.getElementById("btnViewSlides").classList.toggle("active", viewName === "slides");
      document.getElementById("btnViewJson").classList.toggle("active", viewName === "json");

      document.getElementById("dashboardView").style.display = viewName === "dashboard" ? "grid" : "none";
      document.getElementById("slidesView").style.display = viewName === "slides" ? "block" : "none";
      document.getElementById("jsonView").style.display = viewName === "json" ? "block" : "none";

      if (viewName === "dashboard" && map) {
        setTimeout(() => map.invalidateSize(), 150);
      }
    }

    // ── SLIDE DECK LOGIC ──
    function changeSlide(direction) {
      let next = currentSlideIdx + direction;
      if (next < 0) next = totalSlides - 1;
      if (next >= totalSlides) next = 0;
      gotoSlide(next);
    }

    function gotoSlide(idx) {
      currentSlideIdx = idx;
      for (let i = 0; i < totalSlides; i++) {
        const slide = document.getElementById(`slide-${i}`);
        if (slide) slide.classList.toggle("active", i === idx);
      }

      const dots = document.querySelectorAll("#slideDots .step-dot");
      dots.forEach((d, i) => d.classList.toggle("active", i === idx));

      document.getElementById("slideCounter").textContent = `Slide ${idx + 1} of ${totalSlides}`;
    }

    function setupKeyboardNav() {
      window.addEventListener("keydown", (e) => {
        if (document.getElementById("slidesView").style.display === "block") {
          if (e.key === "ArrowRight" || e.key === " ") {
            changeSlide(1);
          } else if (e.key === "ArrowLeft") {
            changeSlide(-1);
          }
        }
      });
    }

    // ── JSON EXPLORER ──
    function loadJsonTab(fileName) {
      activeJsonFile = fileName;
      const buttons = document.querySelectorAll(".json-tab-btn");
      buttons.forEach(btn => {
        btn.classList.toggle("active", btn.textContent.includes(fileName.split("/").pop()));
      });

      const viewer = document.getElementById("jsonCodeViewer");
      viewer.textContent = `Loading ${fileName}...`;

      fetch(`/api/json?case=${encodeURIComponent(currentCaseId)}&file=${encodeURIComponent(fileName)}`)
        .then(res => {
          if (!res.ok) throw new Error("File not found");
          return res.json();
        })
        .then(data => {
          viewer.textContent = JSON.stringify(data, null, 2);
        })
        .catch(err => {
          if (fileName === "case_full.json" && currentCaseData) {
            viewer.textContent = JSON.stringify(currentCaseData, null, 2);
          } else if (fileName.includes("forecast") && currentCaseData && currentCaseData.forecast) {
            viewer.textContent = JSON.stringify(currentCaseData.forecast, null, 2);
          } else {
            viewer.textContent = `// Note: '${fileName}' not directly found for ${currentCaseId}.\n// Showing merged dossier payload:\n` + JSON.stringify(currentCaseData, null, 2);
          }
        });
    }

    function copyJsonToClipboard() {
      const viewer = document.getElementById("jsonCodeViewer");
      navigator.clipboard.writeText(viewer.textContent)
        .then(() => alert("JSON copied to clipboard!"))
        .catch(err => console.error(err));
    }

    function downloadCurrentJson() {
      const viewer = document.getElementById("jsonCodeViewer");
      const blob = new Blob([viewer.textContent], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${currentCaseId}_${activeJsonFile.replace('/', '_')}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  </script>
</body>
</html>
"""


class OilSpillAppHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP Request Handler serving frontend and REST endpoints."""

    def log_message(self, format, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. Frontend UI
        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))
            return

        # 2. API: Cases
        elif path == "/api/cases":
            cases = discover_cases()
            self.send_json_response(cases)
            return

        # 3. API: Case Details
        elif path == "/api/case":
            case_id = query.get("id", ["CASE a1"])[0]
            data = load_case_data(case_id)
            self.send_json_response(data)
            return

        # 4. API: Metrics
        elif path == "/api/metrics":
            metrics = load_ml_metrics()
            self.send_json_response(metrics)
            return

        # 5. API: Raw JSON Files
        elif path == "/api/json":
            case_id = query.get("case", ["CASE a1"])[0]
            file_rel = query.get("file", ["case_full.json"])[0]

            if ".." in file_rel or file_rel.startswith("/"):
                self.send_error(400, "Invalid file path")
                return

            found_file = None
            for parent in ["CASES", "cases"]:
                candidate1 = WORKSPACE_ROOT / parent / case_id / file_rel
                candidate2 = WORKSPACE_ROOT / parent / case_id.replace("_", " ") / file_rel
                if candidate1.exists():
                    found_file = candidate1
                    break
                elif candidate2.exists():
                    found_file = candidate2
                    break

            if found_file and found_file.is_file():
                try:
                    with open(found_file, "r", encoding="utf-8") as f:
                        content = json.load(f)
                    self.send_json_response(content)
                    return
                except Exception as e:
                    self.send_error(500, f"Error reading file: {str(e)}")
                    return
            else:
                self.send_error(404, f"File '{file_rel}' not found for {case_id}")
                return

        # 6. API: Image Server
        elif path == "/api/image":
            case_id = query.get("case", ["CASE a1"])[0]
            name_hint = query.get("name", ["preview"])[0]

            found_img = None
            for parent in ["CASES", "cases"]:
                for sub in [case_id, case_id.replace("_", " ")]:
                    cdir = WORKSPACE_ROOT / parent / sub
                    if cdir.is_dir():
                        if name_hint == "preview":
                            for img in cdir.glob("*spill_preview.png"):
                                found_img = img
                                break
                        elif name_hint == "overlay":
                            for img in cdir.glob("*filtered_overlay.png"):
                                found_img = img
                                break
                        if not found_img:
                            for img in cdir.glob("*.png"):
                                found_img = img
                                break
                    if found_img:
                        break
                if found_img:
                    break

            if found_img and found_img.exists():
                try:
                    with open(found_img, "rb") as f:
                        img_bytes = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(img_bytes)))
                    self.send_header("Cache-Control", "public, max-age=3600")
                    self.end_headers()
                    self.wfile.write(img_bytes)
                    return
                except Exception as e:
                    self.send_error(500, f"Image read error: {str(e)}")
                    return
            else:
                self.send_error(404, "Image not found")
                return

        self.send_error(404, "Endpoint not found")

    def send_json_response(self, obj):
        data_bytes = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data_bytes)


def main():
    import argparse
    env_port = os.environ.get("PORT")
    default_port = int(env_port) if env_port else 5000
    parser = argparse.ArgumentParser(description="SIH26143 Arabian Sea Spill Localhost App")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to bind (default: {default_port})")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the default web browser")
    args = parser.parse_args()

    port = args.port if env_port else find_free_port(args.port)
    host = "0.0.0.0"

    server_address = (host, port)
    httpd = HTTPServer(server_address, OilSpillAppHandler)

    app_url = f"http://localhost:{port}"

    print("=" * 85)
    print(" SIH26143 — ARABIAN SEA OIL SPILL DETECTION & ATTRIBUTION SYSTEM")
    print("=" * 85)
    print(f" [OK] Local Server successfully launched in White & Black Theme!")
    print(f" [URL] Open Dashboard at: {app_url}")
    print(f" [PORT] Bound to: {port}")
    print("=" * 85)
    print(" Features:")
    print("   * Dual-Mode White & Black Theme (1-Click Toggle button in navbar)")
    print("   * Perfect Stage 1 to 7 Visibility with Quick-Filter Stepper Bar")
    print("   * High-contrast mathematical cards (zero raw LaTeX artifacts)")
    print("   * Integrated Map API Token & Leaflet Layer Controller")
    print("=" * 85)
    print(" Press Ctrl+C to stop the server.\n")

    if not args.no_browser and not env_port:
        try:
            webbrowser.open_new_tab(app_url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped gracefully by user.")
        httpd.server_close()


if __name__ == "__main__":
    main()
