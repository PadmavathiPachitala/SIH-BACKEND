"""
stage5_hindcast.py — Stage 5: Hindcasting (Origin Estimation) for SIH26143

Back-tracks oil spill particle clouds backwards in time (24h, 48h, 72h) from detection centroid
using ocean surface currents (and optional wind drift) from NetCDF datasets.

Optimized with in-memory dataset loading and vectorized spatial sampling.

Saves per-case outputs:
- cases/<case_id>/stage5/hindcast.json
- cases/<case_id>/stage5/trajectory_24h.geojson
- cases/<case_id>/stage5/trajectory_48h.geojson
- cases/<case_id>/stage5/trajectory_72h.geojson
- cases/<case_id>/stage5/origins.geojson

Saves combined summary:
- cases/stage5_summary.json

Next-step context:
Stage 5 outputs estimated origins + trajectories for Stage 6 vessel AIS ranking.
"""

import os
import sys
import glob
import json
import math
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

# ── HINDCAST CONFIGURATION ───────────────────────────────────────────────────
DURATIONS_HOURS   = [24, 48, 72]
TIME_STEP_HOURS   = 6
WIND_DRIFT_FACTOR = 0.03
N_PARTICLES       = 30
SEED_RADIUS_KM    = 1.0   # Seed radius around centroid in km

DEFAULT_DETECTION_TIMES = {
    "CASE a1": "2026-07-10T00:48:00Z",
    "CASE a2": "2026-08-14T00:57:00Z",
    "case_a1": "2026-07-10T00:48:00Z",
    "case_a2": "2026-08-14T00:57:00Z"
}
# ─────────────────────────────────────────────────────────────────────────────


def find_case_folders(base_dirs=None):
    """Finds case folders under cases/ or CASES/ containing spill.json."""
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

            spill_json = os.path.join(full_path, "spill.json")
            if not os.path.exists(spill_json):
                spill_json_upper = os.path.join(full_path, "SPILL.json")
                if os.path.exists(spill_json_upper):
                    spill_json = spill_json_upper
                else:
                    continue

            seen.add(real_path)
            case_dirs.append((entry, full_path, spill_json))

    return case_dirs


def find_netcdf_for_case(case_id, case_dir):
    """Searches for relevant Stage 4 NetCDF dataset files automatically across search paths."""
    c_lower = case_id.lower().replace(" ", "_")
    c_dir_basename = os.path.basename(case_dir)

    search_patterns = [
        os.path.join("data", "stage4", c_lower, "*.nc"),
        os.path.join("data", "stage4", c_dir_basename, "*.nc"),
        os.path.join("data", "STGAE 4", c_dir_basename, "*.nc"),
        os.path.join("data", "STGAE 4", "RAW", "*.nc"),
        os.path.join("data", "STGAE 4", "raw", "*.nc"),
        os.path.join("data", "stage4", "raw", "*.nc"),
        os.path.join("data", "stage4", "*.nc"),
        os.path.join("data", "STGAE 4", "*.nc"),
        os.path.join(case_dir, "*.nc"),
        os.path.join("data", "**", "*.nc")
    ]

    for pat in search_patterns:
        matches = sorted(glob.glob(pat, recursive=True))
        if matches:
            return matches[0]

    return None


def extract_netcdf_mapping(ds):
    """Auto-detects variable and coordinate names in an xarray Dataset."""
    lat_name = None
    for name in ["latitude", "lat", "y", "LATITUDE", "LAT"]:
        if name in ds.coords or name in ds.data_vars:
            lat_name = name
            break

    lon_name = None
    for name in ["longitude", "lon", "x", "LONGITUDE", "LON"]:
        if name in ds.coords or name in ds.data_vars:
            lon_name = name
            break

    time_name = None
    for name in ["time", "t", "TIME", "time_counter"]:
        if name in ds.coords or name in ds.data_vars:
            time_name = name
            break

    u_var = None
    for name in ["uo", "u", "eastward_sea_water_velocity", "surface_eastward_sea_water_velocity", "eastward", "u_curr", "cur_u"]:
        if name in ds.data_vars:
            u_var = name
            break

    v_var = None
    for name in ["vo", "v", "northward_sea_water_velocity", "surface_northward_sea_water_velocity", "northward", "v_curr", "cur_v"]:
        if name in ds.data_vars:
            v_var = name
            break

    u_wind = None
    for name in ["u10", "wind_u", "eastward_wind", "u_wind", "wind10u"]:
        if name in ds.data_vars:
            u_wind = name
            break

    v_wind = None
    for name in ["v10", "wind_v", "northward_wind", "v_wind", "wind10v"]:
        if name in ds.data_vars:
            v_wind = name
            break

    return {
        "lat_name": lat_name,
        "lon_name": lon_name,
        "time_name": time_name,
        "u_var": u_var,
        "v_var": v_var,
        "u_wind": u_wind,
        "v_wind": v_wind
    }


def sample_velocity_vectorized(ds, var_map, lats, lons, target_dt=None):
    """
    Samples u, v current (and optional wind) for an array of particle (lats, lons) at target_dt
    using fast C-level xarray indexing.
    """
    lat_n = var_map["lat_name"]
    lon_n = var_map["lon_name"]
    time_n = var_map["time_name"]
    u_v = var_map["u_var"]
    v_v = var_map["v_var"]
    uw_v = var_map["u_wind"]
    vw_v = var_map["v_wind"]

    ds_lons = ds[lon_n].values
    if np.any(ds_lons > 180.0):
        sample_lons = np.where(lons < 0, lons % 360.0, lons)
    elif np.all(ds_lons <= 180.0):
        sample_lons = np.where(lons > 180.0, (lons + 180.0) % 360.0 - 180.0, lons)
    else:
        sample_lons = lons

    lats_da = xr.DataArray(lats, dims="p")
    lons_da = xr.DataArray(sample_lons, dims="p")

    sel_dict = {lat_n: lats_da, lon_n: lons_da}
    if time_n and time_n in ds.coords and target_dt is not None:
        sel_dict[time_n] = np.datetime64(target_dt.strftime("%Y-%m-%dT%H:%M:%S"))

    try:
        ds_pts = ds.sel(sel_dict, method="nearest")

        if "depth" in ds_pts.dims:
            ds_pts = ds_pts.isel(depth=0)
        elif "deptho" in ds_pts.dims:
            ds_pts = ds_pts.isel(deptho=0)

        u_vals = ds_pts[u_v].values.astype(np.float64)
        v_vals = ds_pts[v_v].values.astype(np.float64)

        valid_mask = ~np.isnan(u_vals) & ~np.isnan(v_vals)
        u_vals = np.nan_to_num(u_vals, nan=0.0)
        v_vals = np.nan_to_num(v_vals, nan=0.0)

        wind_drift = WIND_DRIFT_FACTOR if (uw_v and vw_v) else 0.0
        if wind_drift > 0:
            uw_vals = ds_pts[uw_v].values.astype(np.float64)
            vw_vals = ds_pts[vw_v].values.astype(np.float64)
            valid_wind = ~np.isnan(uw_vals) & ~np.isnan(vw_vals)
            uw_vals = np.nan_to_num(uw_vals, nan=0.0)
            vw_vals = np.nan_to_num(vw_vals, nan=0.0)
            u_vals += np.where(valid_wind, wind_drift * uw_vals, 0.0)
            v_vals += np.where(valid_wind, wind_drift * vw_vals, 0.0)

        return u_vals, v_vals, valid_mask
    except Exception:
        return np.zeros_like(lats), np.zeros_like(lats), np.zeros(len(lats), dtype=bool)


def run_particle_hindcast(ds, var_map, cent_lat, cent_lon, det_datetime, duration_hours):
    """
    Runs Lagrangian backward particle tracking for a given duration.
    """
    np.random.seed(42)  # Deterministic seed

    n_steps = int(duration_hours // TIME_STEP_HOURS)
    dt_sec = TIME_STEP_HOURS * 3600.0

    particles_lat = np.zeros(N_PARTICLES, dtype=np.float64)
    particles_lon = np.zeros(N_PARTICLES, dtype=np.float64)
    valid_flags = np.ones(N_PARTICLES, dtype=bool)

    lat_m_per_deg = 111132.9
    lon_m_per_deg = 111412.8 * math.cos(math.radians(cent_lat))

    for i in range(N_PARTICLES):
        r_km = SEED_RADIUS_KM * math.sqrt(np.random.rand())
        theta = 2.0 * math.pi * np.random.rand()
        d_lat = (r_km * math.cos(theta) * 1000.0) / lat_m_per_deg
        d_lon = (r_km * math.sin(theta) * 1000.0) / lon_m_per_deg
        particles_lat[i] = cent_lat + d_lat
        particles_lon[i] = cent_lon + d_lon

    trajectory = [{
        "lat": round(float(cent_lat), 6),
        "lon": round(float(cent_lon), 6),
        "hours_back": 0
    }]

    current_lats = particles_lat.copy()
    current_lons = particles_lon.copy()

    for step in range(1, n_steps + 1):
        hours_back = step * TIME_STEP_HOURS
        step_dt = det_datetime - timedelta(hours=hours_back)

        step_u, step_v, valid_mask = sample_velocity_vectorized(ds, var_map, current_lats, current_lons, step_dt)
        valid_flags = valid_flags & valid_mask

        # Backward advection update: position_new = position - velocity * dt
        dx = step_u * dt_sec
        dy = step_v * dt_sec

        d_lats = dy / lat_m_per_deg
        d_lons = dx / lon_m_per_deg

        current_lats -= d_lats
        current_lons -= d_lons

        mean_lat = float(np.mean(current_lats))
        mean_lon = float(np.mean(current_lons))
        trajectory.append({
            "lat": round(mean_lat, 6),
            "lon": round(mean_lon, 6),
            "hours_back": hours_back
        })

    origin_lat = round(float(np.mean(current_lats)), 6)
    origin_lon = round(float(np.mean(current_lons)), 6)

    # Particle cloud spread std (km)
    d_lats_km = (current_lats - origin_lat) * 111.13
    d_lons_km = (current_lons - origin_lon) * 111.41 * math.cos(math.radians(origin_lat))
    dists_km = np.sqrt(d_lats_km**2 + d_lons_km**2)
    std_km = round(float(np.std(dists_km)), 4)

    # Confidence Score (0.0 to 1.0)
    c_spread = max(0.0, 1.0 - (std_km / 15.0))
    c_valid = float(np.mean(valid_flags))
    confidence = round(float(np.clip(0.6 * c_spread + 0.4 * c_valid, 0.0, 1.0)), 4)

    return {
        "duration_hours": duration_hours,
        "origin": {"lat": origin_lat, "lon": origin_lon},
        "confidence": confidence,
        "particle_std_km": std_km,
        "trajectory": trajectory
    }


def make_trajectory_geojson(case_id, duration_hours, det_pt, det_time, origin_pt, confidence, trajectory):
    """Builds a FeatureCollection GeoJSON for a single simulation trajectory."""
    coords = [[t["lon"], t["lat"]] for t in trajectory]

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "case_id": case_id,
                "duration_hours": duration_hours,
                "confidence": confidence,
                "type": "advection_trajectory"
            }
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [det_pt["lon"], det_pt["lat"]]},
            "properties": {
                "name": f"Detection Point ({case_id})",
                "detection_time": det_time,
                "type": "detection_point"
            }
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [origin_pt["lon"], origin_pt["lat"]]},
            "properties": {
                "name": f"Estimated Origin ({duration_hours}h)",
                "duration_hours": duration_hours,
                "confidence": confidence,
                "type": "estimated_origin"
            }
        }
    ]

    return {"type": "FeatureCollection", "features": features}


def make_origins_geojson(case_id, det_pt, simulations):
    """Builds a FeatureCollection GeoJSON containing detection point and estimated origins for all durations."""
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [det_pt["lon"], det_pt["lat"]]},
            "properties": {
                "name": f"Detection Point ({case_id})",
                "type": "detection_point"
            }
        }
    ]

    for sim in simulations:
        dur = sim["duration_hours"]
        ori = sim["origin"]
        conf = sim["confidence"]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [ori["lon"], ori["lat"]]},
            "properties": {
                "name": f"Estimated Origin ({dur}h)",
                "duration_hours": dur,
                "confidence": conf,
                "particle_std_km": sim["particle_std_km"],
                "type": "estimated_origin"
            }
        })

    return {"type": "FeatureCollection", "features": features}


def process_case(case_id, case_dir, spill_json_path):
    """Processes a single case through Stage 5 hindcasting."""
    with open(spill_json_path, "r") as f:
        spill_data = json.load(f)

    cent_lat = spill_data.get("centroid_lat")
    cent_lon = spill_data.get("centroid_lon")

    if cent_lat is None or cent_lon is None or spill_data.get("status") == "empty_mask":
        print(f"Warning: Case '{case_id}' has empty mask or missing centroid. Skipping.", file=sys.stderr, flush=True)
        return None

    det_time_str = spill_data.get("detection_time")
    if not det_time_str:
        det_time_str = DEFAULT_DETECTION_TIMES.get(case_id, DEFAULT_DETECTION_TIMES.get(case_id.lower(), "2026-07-10T00:48:00Z"))

    det_time_clean = det_time_str.replace("Z", "+00:00")
    try:
        det_datetime = datetime.fromisoformat(det_time_clean)
    except Exception:
        det_datetime = datetime(2026, 7, 10, 0, 48, tzinfo=timezone.utc)

    nc_path = find_netcdf_for_case(case_id, case_dir)
    if not nc_path:
        print(f"\nError: No Stage 4 NetCDF dataset found for case '{case_id}'.", file=sys.stderr, flush=True)
        print("Expected paths: data/stage4/case_a1/*.nc, data/STGAE 4/RAW/*.nc, etc.", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"\nProcessing case [{case_id}] using NetCDF: {nc_path}", flush=True)

    with xr.open_dataset(nc_path) as raw_ds:
        # Load NetCDF dataset fully into RAM for instant index sampling
        ds = raw_ds.load()
        var_map = extract_netcdf_mapping(ds)

        if not var_map["u_var"] or not var_map["v_var"]:
            print(f"Error: NetCDF dataset '{nc_path}' does not contain recognized u/v current variables.", file=sys.stderr, flush=True)
            print(f"Found variables: {list(ds.data_vars.keys())}", file=sys.stderr, flush=True)
            sys.exit(1)

        simulations = []
        for dur in DURATIONS_HOURS:
            sim = run_particle_hindcast(ds, var_map, cent_lat, cent_lon, det_datetime, dur)
            simulations.append(sim)

    # Select best estimate (highest confidence; prefer 48h if close)
    best_sim = max(simulations, key=lambda s: (s["confidence"], 1 if s["duration_hours"] == 48 else 0))

    best_estimate = {
        "duration_hours": best_sim["duration_hours"],
        "origin": best_sim["origin"],
        "confidence": best_sim["confidence"]
    }

    stage5_out_dir = os.path.join(case_dir, "stage5")
    os.makedirs(stage5_out_dir, exist_ok=True)

    det_pt = {"lat": cent_lat, "lon": cent_lon}
    hindcast_json = {
        "case_id": case_id,
        "detection_point": det_pt,
        "detection_time": det_time_str,
        "settings": {
            "time_step_hours": TIME_STEP_HOURS,
            "wind_drift_factor": WIND_DRIFT_FACTOR if (var_map["u_wind"] and var_map["v_wind"]) else 0.0
        },
        "simulations": simulations,
        "best_estimate": best_estimate
    }

    hindcast_json_path = os.path.join(stage5_out_dir, "hindcast.json")
    with open(hindcast_json_path, "w") as f:
        json.dump(hindcast_json, f, indent=2)

    for sim in simulations:
        dur = sim["duration_hours"]
        traj_geojson = make_trajectory_geojson(
            case_id, dur, det_pt, det_time_str, sim["origin"], sim["confidence"], sim["trajectory"]
        )
        traj_path = os.path.join(stage5_out_dir, f"trajectory_{dur}h.geojson")
        with open(traj_path, "w") as f:
            json.dump(traj_geojson, f, indent=2)

    origins_geojson = make_origins_geojson(case_id, det_pt, simulations)
    origins_path = os.path.join(stage5_out_dir, "origins.geojson")
    with open(origins_path, "w") as f:
        json.dump(origins_geojson, f, indent=2)

    return hindcast_json


def main():
    print("=" * 85, flush=True)
    print(" SIH26143 — STAGE 5: HINDCASTING (Origin Estimation)", flush=True)
    print("=" * 85, flush=True)

    case_dirs = find_case_folders()
    if not case_dirs:
        print("Error: No case folders with spill.json found under 'cases/' or 'CASES/'.", flush=True)
        sys.exit(1)

    print(f"Found {len(case_dirs)} case folder(s) to process:", flush=True)
    for cid, cdir, sjson in case_dirs:
        print(f"  - [{cid}] in '{cdir}' using '{os.path.basename(sjson)}'", flush=True)
    print("-" * 85, flush=True)

    summary_results = []
    for case_id, case_dir, spill_json_path in case_dirs:
        res = process_case(case_id, case_dir, spill_json_path)
        if res:
            summary_results.append(res)

    cases_root = "cases" if os.path.exists("cases") else "CASES"
    summary_path = os.path.join(cases_root, "stage5_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    print("\n" + "=" * 85, flush=True)
    print(" STAGE 5 HINDCASTING SUMMARY TABLE", flush=True)
    print("=" * 85, flush=True)
    header = f"{'case_id':<18} | {'best_duration':<14} | {'origin_lat':<12} | {'origin_lon':<12} | {'confidence':<10}"
    print(header, flush=True)
    print("-" * 85, flush=True)
    for s in summary_results:
        cid = s["case_id"][:17]
        best = s["best_estimate"]
        dur = f"{best['duration_hours']}h"
        lat = f"{best['origin']['lat']:.6f}"
        lon = f"{best['origin']['lon']:.6f}"
        conf = f"{best['confidence']:.4f}"
        print(f"{cid:<18} | {dur:<14} | {lat:<12} | {lon:<12} | {conf:<10}", flush=True)
    print("=" * 85, flush=True)
    print(f"Summary JSON saved to: {summary_path}", flush=True)
    print("=" * 85, flush=True)

    print("\nStage 5 outputs estimated origins + trajectories for Stage 6 vessel AIS ranking.", flush=True)


if __name__ == "__main__":
    main()
