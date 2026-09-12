"""
stage6_ais_demo.py — Stage 6: Demo AIS & Real Filters for SIH26143

Generates reproducible synthetic demo AIS tracks centered around Stage 5 estimated origins,
applies real spatial, temporal, speed, dwell, and ship-type filters, and outputs:
- cases/<case>/stage6/ais_all.json
- cases/<case>/stage6/ais_filtered.json
- cases/<case>/stage6/vessel_tracks.geojson
- cases/stage6_summary.json
- data/ais/demo/<case>_demo.csv

Data source tag: "synthetic_demo_ais"
"""

import os
import sys
import glob
import json
import math
import random
import numpy as np
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

# ── DEFAULT FILTER CONSTANTS ──────────────────────────────────────────────────
SEARCH_RADIUS_KM       = 30.0
TIME_WINDOW_HOURS      = 12.0
MAX_SOG_NEAR_ORIGIN_KN = 6.0
NEAR_ORIGIN_RADIUS_KM  = 15.0
MIN_DWELL_MINUTES      = 20.0
ALLOWED_SHIP_TYPES     = ["Oil Tanker", "Product Tanker", "Cargo", "Tanker"]

DEFAULT_DETECTION_TIMES = {
    "CASE a1": "2026-07-10T00:48:00Z",
    "CASE a2": "2026-08-14T00:57:00Z",
    "CASE a3": "2026-08-02T00:57:00Z",
    "CASE 3a": "2026-08-02T00:57:00Z",
    "case_a1": "2026-07-10T00:48:00Z",
    "case_a2": "2026-08-14T00:57:00Z",
    "case_a3": "2026-08-02T00:57:00Z",
    "case_3a": "2026-08-02T00:57:00Z"
}
# ─────────────────────────────────────────────────────────────────────────────


def haversine_km(lat1, lon1, lat2, lon2):
    """Computes great-circle distance between two (lat, lon) points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def find_case_folders(base_dirs=None):
    """Finds case folders under cases/ or CASES/ containing stage5/hindcast.json."""
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

            hindcast_json = os.path.join(full_path, "stage5", "hindcast.json")
            if not os.path.exists(hindcast_json):
                continue

            seen.add(real_path)
            case_dirs.append((entry, full_path, hindcast_json))

    return case_dirs


def generate_vessel_track(origin_lat, origin_lon, rel_dt, vessel_config):
    """
    Generates a realistic 28-hour AIS trajectory around release_time (t_rel ± 14h)
    sampled every 15 minutes.
    """
    target_closest_km = vessel_config["target_closest_km"]
    base_sog = vessel_config["base_sog"]
    cog_deg = vessel_config["cog_deg"]
    heading_rad = math.radians(cog_deg)

    # Calculate closest approach offset from origin
    offset_angle_rad = math.radians(vessel_config.get("offset_angle_deg", 45.0))
    d_lat_deg = (target_closest_km * math.cos(offset_angle_rad)) / 111.13
    d_lon_deg = (target_closest_km * math.sin(offset_angle_rad)) / (111.41 * math.cos(math.radians(origin_lat)))

    closest_pt_lat = origin_lat + d_lat_deg
    closest_pt_lon = origin_lon + d_lon_deg

    # Time steps from -14h to +14h relative to release_time
    start_dt = rel_dt - timedelta(hours=14)
    end_dt = rel_dt + timedelta(hours=14)
    step_minutes = 15

    track_points = []
    curr_dt = start_dt

    while curr_dt <= end_dt:
        dt_hours = (curr_dt - rel_dt).total_seconds() / 3600.0

        # Adjust SOG near origin if vessel is designed to slow down/dwell
        if abs(dt_hours) <= 3.0 and vessel_config.get("slows_near_origin", False):
            sog_kn = vessel_config.get("slow_sog", 2.2)
        else:
            sog_kn = base_sog

        # Add small speed fluctuation
        sog_kn = max(0.5, sog_kn + random.uniform(-0.3, 0.3))

        # Position offset along heading direction relative to closest approach
        dist_offset_nm = sog_kn * dt_hours
        dist_offset_km = dist_offset_nm * 1.852

        pt_lat = closest_pt_lat + (dist_offset_km * math.cos(heading_rad)) / 111.13
        pt_lon = closest_pt_lon + (dist_offset_km * math.sin(heading_rad)) / (111.41 * math.cos(math.radians(origin_lat)))

        # Add small random GPS noise (10-30 meters)
        pt_lat += random.uniform(-0.0002, 0.0002)
        pt_lon += random.uniform(-0.0002, 0.0002)

        dist_to_orig = haversine_km(pt_lat, pt_lon, origin_lat, origin_lon)

        track_points.append({
            "lat": round(pt_lat, 6),
            "lon": round(pt_lon, 6),
            "time_iso": curr_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sog_kn": round(sog_kn, 2),
            "cog_deg": round(cog_deg, 1),
            "distance_to_origin_km": round(dist_to_orig, 2)
        })

        curr_dt += timedelta(minutes=step_minutes)

    return track_points


def evaluate_vessel(vessel_meta, track_points, origin_lat, origin_lon, rel_dt):
    """
    Computes vessel metrics and evaluates boolean filter rules.
    """
    dists = [pt["distance_to_origin_km"] for pt in track_points]
    closest_dist = min(dists)
    closest_idx = dists.index(closest_dist)
    time_at_closest = track_points[closest_idx]["time_iso"]

    # Points inside 15 km near-origin radius
    near_pts = [pt for pt in track_points if pt["distance_to_origin_km"] <= NEAR_ORIGIN_RADIUS_KM]

    if near_pts:
        avg_sog_near = round(float(np.mean([pt["sog_kn"] for pt in near_pts])), 2)
        # Dwell time in minutes = count of 15-min interval points inside 15 km
        dwell_minutes = float(len(near_pts) * 15.0)
    else:
        avg_sog_near = None
        dwell_minutes = 0.0

    # Points within ± TIME_WINDOW_HOURS (12h) of estimated release time
    window_start = rel_dt - timedelta(hours=TIME_WINDOW_HOURS)
    window_end = rel_dt + timedelta(hours=TIME_WINDOW_HOURS)

    pts_in_window = 0
    for pt in track_points:
        pt_dt = datetime.fromisoformat(pt["time_iso"].replace("Z", "+00:00"))
        if window_start <= pt_dt <= window_end:
            pts_in_window += 1

    # Filter Rules
    passes_dist = (closest_dist <= SEARCH_RADIUS_KM)
    passes_time = (pts_in_window > 0)
    passes_speed = (avg_sog_near is not None and avg_sog_near <= MAX_SOG_NEAR_ORIGIN_KN)
    passes_dwell = (dwell_minutes >= MIN_DWELL_MINUTES)
    passes_type = (vessel_meta["ship_type"] in ALLOWED_SHIP_TYPES)

    passes_all = bool(passes_dist and passes_time and passes_speed and passes_dwell and passes_type)

    return {
        "mmsi": vessel_meta["mmsi"],
        "name": vessel_meta["name"],
        "ship_type": vessel_meta["ship_type"],
        "closest_distance_km": round(closest_dist, 2),
        "time_at_closest": time_at_closest,
        "avg_sog_near_origin": avg_sog_near,
        "dwell_minutes_near_origin": dwell_minutes,
        "points_in_time_window": pts_in_window,
        "passes_distance_filter": passes_dist,
        "passes_time_filter": passes_time,
        "passes_speed_filter": passes_speed,
        "passes_dwell_filter": passes_dwell,
        "passes_type_filter": passes_type,
        "passes_all_default_filters": passes_all,
        "track": track_points
    }


def process_case_stage6(case_id, case_dir, hindcast_json_path):
    """Generates synthetic demo vessels and applies real AIS filters for a case."""
    random.seed(42)  # Reproducible demo vessel generation

    with open(hindcast_json_path, "r") as f:
        hindcast_data = json.load(f)

    best_est = hindcast_data.get("best_estimate", {})
    origin = best_est.get("origin", {})
    origin_lat = origin.get("lat")
    origin_lon = origin.get("lon")
    dur_hours = best_est.get("duration_hours", 48)

    det_time_str = hindcast_data.get("detection_time")
    if not det_time_str:
        det_time_str = DEFAULT_DETECTION_TIMES.get(case_id, "2026-07-10T00:48:00Z")

    det_dt = datetime.fromisoformat(det_time_str.replace("Z", "+00:00"))
    rel_dt = det_dt - timedelta(hours=dur_hours)
    rel_time_str = rel_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Define 8 unique vessels per case
    if "a1" in case_id.lower():
        vessel_configs = [
            {"mmsi": 419001001, "name": "DESERT STAR", "ship_type": "Oil Tanker", "target_closest_km": 3.2, "base_sog": 2.5, "cog_deg": 135.0, "slows_near_origin": True, "slow_sog": 2.0, "offset_angle_deg": 30.0},
            {"mmsi": 419001002, "name": "ARABIAN PEARL", "ship_type": "Product Tanker", "target_closest_km": 8.5, "base_sog": 8.0, "cog_deg": 220.0, "slows_near_origin": True, "slow_sog": 3.8, "offset_angle_deg": 120.0},
            {"mmsi": 419001003, "name": "GULF EXPRESS", "ship_type": "Cargo", "target_closest_km": 14.0, "base_sog": 11.5, "cog_deg": 45.0, "slows_near_origin": False, "offset_angle_deg": 210.0},
            {"mmsi": 419001004, "name": "OCEAN PIONEER", "ship_type": "Cargo", "target_closest_km": 19.5, "base_sog": 15.0, "cog_deg": 310.0, "slows_near_origin": False, "offset_angle_deg": 315.0},
            {"mmsi": 419001005, "name": "AL-AMAL 7", "ship_type": "Fishing", "target_closest_km": 18.0, "base_sog": 4.5, "cog_deg": 90.0, "slows_near_origin": True, "slow_sog": 3.5, "offset_angle_deg": 160.0},
            {"mmsi": 419001006, "name": "ASIAN CLIPPER", "ship_type": "Cargo", "target_closest_km": 27.5, "base_sog": 16.5, "cog_deg": 180.0, "slows_near_origin": False, "offset_angle_deg": 280.0},
            {"mmsi": 419001007, "name": "MARITIMA ONE", "ship_type": "Oil Tanker", "target_closest_km": 6.4, "base_sog": 7.0, "cog_deg": 15.0, "slows_near_origin": True, "slow_sog": 3.2, "offset_angle_deg": 75.0},
            {"mmsi": 419001008, "name": "ROYAL PRINCESS", "ship_type": "Passenger", "target_closest_km": 24.0, "base_sog": 18.0, "cog_deg": 270.0, "slows_near_origin": False, "offset_angle_deg": 340.0}
        ]
    else:
        vessel_configs = [
            {"mmsi": 419002001, "name": "SEA CHAMPION", "ship_type": "Oil Tanker", "target_closest_km": 2.85, "base_sog": 2.8, "cog_deg": 140.0, "slows_near_origin": True, "slow_sog": 1.8, "offset_angle_deg": 45.0},
            {"mmsi": 419002002, "name": "INDUS GLORY", "ship_type": "Product Tanker", "target_closest_km": 7.8, "base_sog": 7.5, "cog_deg": 200.0, "slows_near_origin": True, "slow_sog": 3.2, "offset_angle_deg": 110.0},
            {"mmsi": 419002003, "name": "LACCADIVE MONARCH", "ship_type": "Cargo", "target_closest_km": 12.5, "base_sog": 12.0, "cog_deg": 60.0, "slows_near_origin": False, "offset_angle_deg": 220.0},
            {"mmsi": 419002004, "name": "MALABAR NAVIGATOR", "ship_type": "Cargo", "target_closest_km": 18.2, "base_sog": 14.5, "cog_deg": 320.0, "slows_near_origin": False, "offset_angle_deg": 300.0},
            {"mmsi": 419002005, "name": "BLUE FIN 3", "ship_type": "Fishing", "target_closest_km": 16.5, "base_sog": 4.0, "cog_deg": 100.0, "slows_near_origin": True, "slow_sog": 3.0, "offset_angle_deg": 150.0},
            {"mmsi": 419002006, "name": "ORIENT RUNNER", "ship_type": "Cargo", "target_closest_km": 26.0, "base_sog": 16.0, "cog_deg": 190.0, "slows_near_origin": False, "offset_angle_deg": 270.0},
            {"mmsi": 419002007, "name": "PETRO CHEVRON", "ship_type": "Oil Tanker", "target_closest_km": 5.9, "base_sog": 6.5, "cog_deg": 30.0, "slows_near_origin": True, "slow_sog": 2.9, "offset_angle_deg": 80.0},
            {"mmsi": 419002008, "name": "ISLAND QUEEN", "ship_type": "Passenger", "target_closest_km": 22.5, "base_sog": 17.5, "cog_deg": 260.0, "slows_near_origin": False, "offset_angle_deg": 330.0}
        ]

    vessels_all = []
    vessels_filtered = []

    for cfg in vessel_configs:
        track = generate_vessel_track(origin_lat, origin_lon, rel_dt, cfg)
        v_data = evaluate_vessel(cfg, track, origin_lat, origin_lon, rel_dt)
        vessels_all.append(v_data)
        if v_data["passes_all_default_filters"]:
            vessels_filtered.append(v_data)

    # 1. Output ais_all.json and ais_filtered.json
    filter_defaults = {
        "search_radius_km": SEARCH_RADIUS_KM,
        "time_window_hours": TIME_WINDOW_HOURS,
        "max_sog_near_origin_kn": MAX_SOG_NEAR_ORIGIN_KN,
        "near_origin_radius_km": NEAR_ORIGIN_RADIUS_KM,
        "min_dwell_minutes": MIN_DWELL_MINUTES
    }

    ais_all_data = {
        "case_id": case_id,
        "data_source": "synthetic_demo_ais",
        "origin": {"lat": origin_lat, "lon": origin_lon},
        "release_time_estimate": rel_time_str,
        "detection_time": det_time_str,
        "filter_defaults": filter_defaults,
        "vessels": vessels_all
    }

    ais_filtered_data = {
        "case_id": case_id,
        "data_source": "synthetic_demo_ais",
        "origin": {"lat": origin_lat, "lon": origin_lon},
        "release_time_estimate": rel_time_str,
        "detection_time": det_time_str,
        "filter_defaults": filter_defaults,
        "vessels": vessels_filtered
    }

    stage6_dir = os.path.join(case_dir, "stage6")
    os.makedirs(stage6_dir, exist_ok=True)

    with open(os.path.join(stage6_dir, "ais_all.json"), "w") as f:
        json.dump(ais_all_data, f, indent=2)

    with open(os.path.join(stage6_dir, "ais_filtered.json"), "w") as f:
        json.dump(ais_filtered_data, f, indent=2)

    # 2. Output vessel_tracks.geojson
    features = []
    for v in vessels_all:
        coords = [[pt["lon"], pt["lat"]] for pt in v["track"]]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "mmsi": v["mmsi"],
                "name": v["name"],
                "ship_type": v["ship_type"],
                "closest_distance_km": v["closest_distance_km"],
                "passes_all_default_filters": v["passes_all_default_filters"]
            }
        })

    geojson_data = {"type": "FeatureCollection", "features": features}
    with open(os.path.join(stage6_dir, "vessel_tracks.geojson"), "w") as f:
        json.dump(geojson_data, f, indent=2)

    # 3. Output demo CSV file in data/ais/demo/
    demo_csv_dir = os.path.join("data", "ais", "demo")
    os.makedirs(demo_csv_dir, exist_ok=True)
    c_clean = case_id.lower().replace(" ", "_")
    csv_path = os.path.join(demo_csv_dir, f"{c_clean}_demo.csv")

    with open(csv_path, "w") as f:
        f.write("mmsi,name,ship_type,timestamp,latitude,longitude,sog,cog,distance_to_origin_km\n")
        for v in vessels_all:
            for pt in v["track"]:
                line = f"{v['mmsi']},{v['name']},{v['ship_type']},{pt['time_iso']},{pt['lat']},{pt['lon']},{pt['sog_kn']},{pt['cog_deg']},{pt['distance_to_origin_km']}\n"
                f.write(line)

    # Compute summary metrics for stage6_summary.json
    tankers = [v for v in vessels_filtered if "tanker" in v["ship_type"].lower()]
    closest_tanker_dist = min([t["closest_distance_km"] for t in tankers]) if tankers else (min([v["closest_distance_km"] for v in vessels_filtered]) if vessels_filtered else None)
    top_3_names = [v["name"] for v in vessels_filtered[:3]]

    return {
        "case_id": case_id,
        "n_all": len(vessels_all),
        "n_filtered": len(vessels_filtered),
        "closest_pass_km": min([v["closest_distance_km"] for v in vessels_filtered]) if vessels_filtered else None,
        "closest_tanker_distance_km": closest_tanker_dist,
        "top_3_filtered_names": top_3_names
    }


def main():
    print("=" * 85, flush=True)
    print(" SIH26143 — STAGE 6: DEMO AIS + REAL FILTERS", flush=True)
    print("=" * 85, flush=True)

    case_dirs = find_case_folders()
    if not case_dirs:
        print("Error: No case folders with stage5/hindcast.json found under 'cases/' or 'CASES/'.", flush=True)
        sys.exit(1)

    print(f"Found {len(case_dirs)} case folder(s) to process:", flush=True)
    for cid, cdir, hjson in case_dirs:
        print(f"  - [{cid}] in '{cdir}' using '{os.path.basename(hjson)}'", flush=True)
    print("-" * 85, flush=True)

    summary_results = []
    for case_id, case_dir, hindcast_json_path in case_dirs:
        res = process_case_stage6(case_id, case_dir, hindcast_json_path)
        summary_results.append(res)

    # Save combined stage6_summary.json
    cases_root = "cases" if os.path.exists("cases") else "CASES"
    summary_path = os.path.join(cases_root, "stage6_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 85, flush=True)
    print(" STAGE 6 DEMO AIS & REAL FILTERS SUMMARY TABLE", flush=True)
    print("=" * 85, flush=True)
    header = f"{'case_id':<18} | {'n_all':<8} | {'n_filtered':<12} | {'closest_pass_km':<18}"
    print(header, flush=True)
    print("-" * 85, flush=True)
    for s in summary_results:
        cid = s["case_id"][:17]
        n_a = s["n_all"]
        n_f = s["n_filtered"]
        cp = f"{s['closest_pass_km']:.2f} km" if s["closest_pass_km"] is not None else "N/A"
        print(f"{cid:<18} | {n_a:<8} | {n_f:<12} | {cp:<18}", flush=True)
    print("=" * 85, flush=True)
    print(f"Summary JSON saved to: {summary_path}", flush=True)
    print("=" * 85, flush=True)

    print("\nStage 6 outputs filtered AIS candidate vessels for Stage 7 suspect ranking.", flush=True)


if __name__ == "__main__":
    main()
