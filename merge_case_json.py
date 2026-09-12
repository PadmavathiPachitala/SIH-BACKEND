# Example: python merge_case_json.py --case_dir "CASES/CASE a1" --out "CASES/CASE a1/case_full.json"
"""
merge_case_json.py — SIH26143 Case JSON Merger

Merges Stage 3 spill.json + Stage 7 ranking/summary + optional Stagpython merge_case_json.py --case_dir "CASES/CASE a1"
e 5 hindcast.json
into one case_full.json for the web dashboard.

CLI Usage:
  python merge_case_json.py --case_dir "CASES/CASE a1" [--out "CASES/CASE a1/case_full.json"] [--case_id "CASE a1"]
"""

import sys
import os
import json
import argparse


def get_first_key(d, keys, default=None):
    """Returns the value for the first matching key found in dictionary d."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def find_spill_file(case_dir, case_id=None):
    """
    Finds Stage 3 spill.json case-insensitively within case_dir or subfolders/parents.
    Returns (dict, filepath) or (None, None).
    """
    candidates = []
    if os.path.exists(case_dir):
        for root, _, files in os.walk(case_dir):
            for f in files:
                f_lower = f.lower()
                if f_lower == "spill.json" or "spill" in f_lower or f_lower.startswith("stage3"):
                    candidates.append(os.path.join(root, f))

    # Priority 1: Exact 'spill.json' or 'SPILL.json'
    for c in candidates:
        if os.path.basename(c).lower() == "spill.json":
            try:
                with open(c, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if isinstance(data, dict):
                        return data, c
                    elif isinstance(data, list):
                        for item in data:
                            if case_id and str(item.get("case_id", "")).lower() == case_id.lower():
                                return item, c
                        if len(data) > 0:
                            return data[0], c
            except Exception:
                pass

    # Priority 2: Other matching files inside case_dir
    for c in candidates:
        try:
            with open(c, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, dict):
                    return data, c
                elif isinstance(data, list):
                    for item in data:
                        if case_id and str(item.get("case_id", "")).lower() == case_id.lower():
                            return item, c
                    if len(data) > 0:
                        return data[0], c
        except Exception:
            pass

    # Priority 3: Parent directory summary files (e.g., CASES/stage3_summary.json)
    norm_case = os.path.normpath(case_dir)
    parent_dir = os.path.dirname(norm_case)
    target_id = case_id or os.path.basename(norm_case)

    if parent_dir and os.path.exists(parent_dir):
        for summary_name in ["stage3_summary.json", "spill_summary.json"]:
            summary_path = os.path.join(parent_dir, summary_name)
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                        if isinstance(data, list):
                            for item in data:
                                item_cid = str(item.get("case_id", "")).lower().replace("_", " ").strip()
                                target_cid = target_id.lower().replace("_", " ").strip()
                                if item_cid == target_cid or target_cid in item_cid:
                                    return item, summary_path
                except Exception:
                    pass

    return None, None


def find_stage7_file(case_dir, case_id=None):
    """
    Finds stage7_summary.json OR ranking.json OR any *stage7*.json / *rank*.json with a suspect list.
    Returns (suspects_list, stage7_dict, filepath) or (None, None, None).
    """
    candidates = []
    if os.path.exists(case_dir):
        for root, _, files in os.walk(case_dir):
            for f in files:
                f_lower = f.lower()
                if (any(t in f_lower for t in ["stage7", "rank", "suspect"])) and f_lower.endswith(".json"):
                    candidates.append(os.path.join(root, f))

    norm_case = os.path.normpath(case_dir)
    target_id = case_id or os.path.basename(norm_case)

    def extract_suspects(data, target_cid):
        if isinstance(data, dict):
            s_list = get_first_key(data, ["ranked_suspects", "suspects", "suspect_vessels", "vessels", "suspect_list"])
            if isinstance(s_list, list):
                return s_list, data
        elif isinstance(data, list):
            target_norm = target_cid.lower().replace("_", " ").strip() if target_cid else ""
            for item in data:
                if isinstance(item, dict):
                    item_cid = str(item.get("case_id", "")).lower().replace("_", " ").strip()
                    if target_norm and (item_cid == target_norm or target_norm in item_cid):
                        s_list = get_first_key(item, ["ranked_suspects", "suspects", "suspect_vessels", "vessels", "suspect_list"])
                        if isinstance(s_list, list):
                            return s_list, item
            # If the list itself contains suspect objects
            if len(data) > 0 and isinstance(data[0], dict) and ("mmsi" in data[0] or "final_score" in data[0] or "name" in data[0]):
                return data, {"ranked_suspects": data}
        return None, None

    # Priority 1: Inside case_dir
    # Sort candidates so ranked_suspects or ranking comes first
    candidates.sort(key=lambda p: (0 if "ranked" in os.path.basename(p).lower() else (1 if "stage7" in os.path.basename(p).lower() else 2)))
    for c in candidates:
        try:
            with open(c, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                s_list, s_dict = extract_suspects(data, target_id)
                if s_list is not None:
                    return s_list, s_dict, c
        except Exception:
            pass

    # Priority 2: Parent dir fallback (e.g. CASES/stage7_summary.json)
    parent_dir = os.path.dirname(norm_case)
    if parent_dir and os.path.exists(parent_dir):
        for summary_name in ["stage7_summary.json", "ranking_summary.json", "stage7_summary_output.json"]:
            summary_path = os.path.join(parent_dir, summary_name)
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                        s_list, s_dict = extract_suspects(data, target_id)
                        if s_list is not None:
                            return s_list, s_dict, summary_path
                except Exception:
                    pass

    return None, None, None


def find_hindcast_file(case_dir, case_id=None):
    """
    Finds Stage 5 hindcast.json within case_dir or subfolders.
    Returns (dict, filepath) or (None, None).
    """
    candidates = []
    if os.path.exists(case_dir):
        for root, _, files in os.walk(case_dir):
            for f in files:
                f_lower = f.lower()
                if ("hindcast" in f_lower or "stage5" in f_lower) and f_lower.endswith(".json"):
                    candidates.append(os.path.join(root, f))

    for c in candidates:
        try:
            with open(c, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, dict):
                    return data, c
                elif isinstance(data, list):
                    for item in data:
                        if case_id and str(item.get("case_id", "")).lower() == case_id.lower():
                            return item, c
                    if len(data) > 0:
                        return data[0], c
        except Exception:
            pass
    return None, None


def merge_case_data(case_dir, out_path=None, case_id_override=None):
    """Performs case merge and outputs unified case_full.json."""
    norm_case_dir = os.path.normpath(case_dir)
    fallback_case_id = os.path.basename(norm_case_dir)

    # 1. Load spill.json (Required)
    spill_data, spill_path = find_spill_file(norm_case_dir, case_id=case_id_override)
    if not spill_data:
        print(f"[ERROR] Missing required input: Stage 3 spill.json in '{case_dir}'", file=sys.stderr)
        sys.exit(1)

    # Resolve case_id
    case_id = (case_id_override or
               get_first_key(spill_data, ["case_id", "case_name", "id"]) or
               fallback_case_id)

    # 2. Load Stage 7 ranking/summary (Required)
    raw_suspects, stage7_data, stage7_path = find_stage7_file(norm_case_dir, case_id=case_id)
    if raw_suspects is None:
        print(f"[ERROR] Missing required input: Stage 7 ranking/summary file with suspect list in '{case_dir}'", file=sys.stderr)
        sys.exit(1)

    # 3. Load optional Stage 5 hindcast.json
    hindcast_data, hindcast_path = find_hindcast_file(norm_case_dir, case_id=case_id)

    # Build Detection block
    centroid_lat = float(get_first_key(spill_data, ["centroid_lat", "lat", "center_lat", "latitude"], default=0.0))
    centroid_lon = float(get_first_key(spill_data, ["centroid_lon", "lon", "center_lon", "longitude"], default=0.0))
    area_km2 = float(get_first_key(spill_data, ["area_km2", "spill_area_km2", "area", "spill_area"], default=0.0))
    perimeter_km = float(get_first_key(spill_data, ["perimeter_km", "perimeter"], default=0.0))
    major_axis_km = float(get_first_key(spill_data, ["major_axis_km", "major_axis"], default=0.0))
    minor_axis_km = float(get_first_key(spill_data, ["minor_axis_km", "minor_axis"], default=0.0))
    orientation_deg = float(get_first_key(spill_data, ["orientation_deg", "orientation"], default=0.0))
    area_pixels = int(get_first_key(spill_data, ["area_pixels", "pixels"], default=0))

    det_time = str(get_first_key(spill_data, ["detection_time", "time", "timestamp", "datetime"],
                                default=get_first_key(stage7_data or {}, ["detection_time", "time", "timestamp"], default="")))

    bbox_latlon = get_first_key(spill_data, ["bbox_latlon", "bbox"], default=[])
    crs = str(get_first_key(spill_data, ["crs"], default="EPSG:4326"))
    mask_path = str(get_first_key(spill_data, ["mask_path"], default=""))

    detection_obj = {
        "lat": centroid_lat,
        "lon": centroid_lon,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "area_km2": round(area_km2, 4),
        "perimeter_km": round(perimeter_km, 4),
        "major_axis_km": round(major_axis_km, 4),
        "minor_axis_km": round(minor_axis_km, 4),
        "orientation_deg": round(orientation_deg, 2),
        "area_pixels": area_pixels,
        "detection_time": det_time,
        "time": det_time,
        "bbox_latlon": bbox_latlon,
        "crs": crs,
        "mask_path": mask_path
    }

    # Process & normalize suspects list
    processed_suspects = []
    for sus in raw_suspects:
        if not isinstance(sus, dict):
            continue

        mmsi = get_first_key(sus, ["mmsi", "vessel_mmsi", "id", "imo"], default=0)
        name = str(get_first_key(sus, ["name", "vessel_name", "ship_name"], default="UNKNOWN VESSEL"))
        ship_type = str(get_first_key(sus, ["ship_type", "vessel_type", "type", "category"], default="Other"))

        # Flexible score extraction
        score_val = float(get_first_key(sus, ["pipeline_score", "final_score", "score", "total_score", "rank_score"], default=0.0))
        proximity_score = float(get_first_key(sus, ["proximity_score", "dist_score", "proximity"], default=0.0))
        dwell_score = float(get_first_key(sus, ["dwell_score", "dwell"], default=0.0))
        speed_score = float(get_first_key(sus, ["speed_score", "speed"], default=0.0))
        path_score = float(get_first_key(sus, ["path_score", "time_score", "trajectory_score", "path"], default=0.0))
        ship_type_score = float(get_first_key(sus, ["ship_type_score", "type_score", "vessel_type_score"], default=0.0))

        # Metrics / feature defaults
        metrics_dict = sus.get("metrics", {}) if isinstance(sus.get("metrics"), dict) else {}
        dist = get_first_key(metrics_dict, ["closest_distance_km", "distance_km", "min_dist_km", "distance"],
                             default=get_first_key(sus, ["closest_distance_km", "distance_km", "min_dist_km", "distance"], default=999.0))
        dwell_m = get_first_key(metrics_dict, ["dwell_minutes", "min_dwell_min", "dwell_min", "dwell"],
                                default=get_first_key(sus, ["dwell_minutes", "min_dwell_min", "dwell_min", "dwell"], default=0.0))
        
        spd_mps = get_first_key(metrics_dict, ["vessel_speed_mps", "speed_mps"],
                                default=get_first_key(sus, ["vessel_speed_mps", "speed_mps"], default=None))
        spd_kn = get_first_key(metrics_dict, ["speed_kn", "vessel_speed_kn", "speed_knots"],
                               default=get_first_key(sus, ["speed_kn", "vessel_speed_kn", "speed_knots"], default=None))

        if spd_kn is None and spd_mps is not None:
            spd_kn = round(float(spd_mps) * 1.94384, 2)
        elif spd_kn is None:
            spd_kn = 0.0
        
        if spd_mps is None and spd_kn is not None:
            spd_mps = round(float(spd_kn) / 1.94384, 2)
        elif spd_mps is None:
            spd_mps = 0.0

        time_off = get_first_key(metrics_dict, ["time_offset_hours", "time_offset", "time_diff_hours"],
                                 default=get_first_key(sus, ["time_offset_hours", "time_offset", "time_diff_hours"], default=0.0))
        
        justification = str(get_first_key(sus, ["justification", "reason", "notes"], default=""))

        suspect_item = {
            "mmsi": mmsi,
            "name": name,
            "ship_type": ship_type,
            "pipeline_score": round(score_val, 4),
            "final_score": round(score_val, 4),
            "proximity_score": round(proximity_score, 4),
            "time_score": round(path_score, 4),
            "path_score": round(path_score, 4),
            "speed_score": round(speed_score, 4),
            "dwell_score": round(dwell_score, 4),
            "type_score": round(ship_type_score, 4),
            "ship_type_score": round(ship_type_score, 4),
            "distance_km": round(float(dist), 2),
            "closest_distance_km": round(float(dist), 2),
            "dwell_minutes": round(float(dwell_m), 1),
            "min_dwell_min": round(float(dwell_m), 1),
            "speed_kn": round(float(spd_kn), 2),
            "vessel_speed_kn": round(float(spd_kn), 2),
            "vessel_speed_mps": round(float(spd_mps), 2),
            "time_offset_hours": round(float(time_off), 2),
            "justification": justification,
            "metrics": {
                "closest_distance_km": round(float(dist), 2),
                "dwell_minutes": round(float(dwell_m), 1),
                "vessel_speed_kn": round(float(spd_kn), 2),
                "vessel_speed_mps": round(float(spd_mps), 2),
                "time_offset_hours": round(float(time_off), 2)
            }
        }
        processed_suspects.append(suspect_item)

    # Sort suspects by pipeline_score desc
    processed_suspects.sort(key=lambda s: s["pipeline_score"], reverse=True)
    for idx, s in enumerate(processed_suspects, start=1):
        s["rank"] = idx

    # If hindcast is missing, attempt to synthesize basic hindcast block from stage7 if available
    if hindcast_data is None and stage7_data:
        est_origin = stage7_data.get("estimated_origin")
        rel_time = stage7_data.get("release_time_estimate")
        if est_origin or rel_time:
            hindcast_data = {
                "detection_point": {"lat": centroid_lat, "lon": centroid_lon},
                "detection_time": det_time,
                "best_estimate": {
                    "origin": est_origin,
                    "release_time": rel_time
                }
            }

    # Build Map block
    bounds = bbox_latlon if bbox_latlon else [centroid_lon - 0.1, centroid_lat - 0.1, centroid_lon + 0.1, centroid_lat + 0.1]
    map_obj = {
        "center": [centroid_lat, centroid_lon],
        "zoom": 10,
        "bounds": bounds,
        "spill_centroid": [centroid_lat, centroid_lon]
    }

    # Filter Defaults (Required)
    filter_defaults = {
        "max_distance_km": 30,
        "min_dwell_min": 0,
        "max_speed_kn": 8,
        "ship_types": ["Tanker", "Cargo", "Passenger", "Other"],
        "time_band": "around",
        "top_n": 10,
        "weights": {
            "proximity": 0.30,
            "dwell": 0.25,
            "speed": 0.20,
            "path": 0.15,
            "ship_type": 0.10
        }
    }

    # Check for optional Stage 3b age.json
    age_file = os.path.join(norm_case_dir, "age.json")
    if not os.path.exists(age_file):
        alt_age = os.path.join(norm_case_dir, "age_estimate.json")
        if os.path.exists(alt_age):
            age_file = alt_age
    age_data = None
    if os.path.exists(age_file):
        try:
            with open(age_file, "r", encoding="utf-8") as fp:
                age_data = json.load(fp)
        except Exception:
            pass

    # Check for optional Stage 5b forecast.json and forecast_envelope.geojson
    forecast_file = os.path.join(norm_case_dir, "forecast.json")
    forecast_data = None
    if os.path.exists(forecast_file):
        try:
            with open(forecast_file, "r", encoding="utf-8") as fp:
                forecast_data = json.load(fp)
        except Exception:
            pass

    forecast_envelope_file = os.path.join(norm_case_dir, "forecast_envelope.geojson")
    forecast_envelope_data = None
    if os.path.exists(forecast_envelope_file):
        try:
            with open(forecast_envelope_file, "r", encoding="utf-8") as fp:
                forecast_envelope_data = json.load(fp)
        except Exception:
            pass

    # Final Output Schema
    case_full = {
        "case_id": case_id,
        "status": str(get_first_key(spill_data, ["status"], default="characterised")),
        "title": str(get_first_key(spill_data, ["title"], default=f"Oil Spill Case - {case_id}")),
        "detection": detection_obj,
        "age": age_data,
        "hindcast": hindcast_data,
        "forecast": forecast_data,
        "forecast_envelope": forecast_envelope_data,
        "suspects": processed_suspects,
        "map": map_obj,
        "filter_defaults": filter_defaults
    }

    # Determine destination output file path
    if not out_path:
        out_path = os.path.join(norm_case_dir, "case_full.json")

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(case_full, fp, indent=2, ensure_ascii=False)

    print(f"[OK] Successfully merged case '{case_id}' -> {out_path} ({len(processed_suspects)} suspects)")

    # Automatically synchronize to my_dashboard_data/
    dash_dir = "my_dashboard_data"
    os.makedirs(dash_dir, exist_ok=True)
    clean_id = case_id.lower().replace(" ", "_")
    dash_filenames = [f"{clean_id}.json"]
    if clean_id == "case_a3":
        dash_filenames.append("case_3a.json")
    elif clean_id == "case_3a":
        dash_filenames.append("case_a3.json")

    for fname in dash_filenames:
        dash_path = os.path.join(dash_dir, fname)
        if os.path.abspath(dash_path) != os.path.abspath(out_path):
            with open(dash_path, "w", encoding="utf-8") as fp:
                json.dump(case_full, fp, indent=2, ensure_ascii=False)
            print(f"[OK] Automatically synced to frontend folder -> {dash_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge Stage 3 spill.json + Stage 7 ranking into case_full.json")
    parser.add_argument("--case_dir", type=str, required=True, help="Path to target case directory")
    parser.add_argument("--out", type=str, default=None, help="Path to output JSON file (default: <case_dir>/case_full.json)")
    parser.add_argument("--case_id", type=str, default=None, help="Override case ID")

    args = parser.parse_args()
    merge_case_data(case_dir=args.case_dir, out_path=args.out, case_id_override=args.case_id)


if __name__ == "__main__":
    main()
