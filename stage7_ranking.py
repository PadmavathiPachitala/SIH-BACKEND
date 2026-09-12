"""
stage7_ranking.py — Stage 7: Suspect Vessel Ranking for SIH26143

Computes 5 normalized attribution criteria (Proximity, Time, Speed Match, Dwell, Ship Type),
calculates weighted final attribution scores, ranks top suspect vessels, outputs justification strings,
saves per-case stage7/ranked_suspects.json, and generates cases/stage7_summary.json.

Next-step context:
Stage 7 outputs final suspect vessel ranking and justification reports for attribution submission.
"""

import os
import sys
import glob
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from tqdm import tqdm


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
    """Finds case folders under cases/ or CASES/ containing stage6 outputs or spill.json."""
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
        if not os.path.exists(base):
            continue
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


def compute_hindcast_drift_speed(det_pt, origin_pt, duration_hours):
    """Computes average hindcast drift speed in m/s from origin to detection point."""
    if not det_pt or not origin_pt or not duration_hours or duration_hours <= 0:
        return 0.5  # Default 0.5 m/s drift scale fallback

    dist_km = haversine_km(origin_pt["lat"], origin_pt["lon"], det_pt["lat"], det_pt["lon"])
    dist_m = dist_km * 1000.0
    time_sec = duration_hours * 3600.0
    drift_mps = dist_m / time_sec
    return round(float(drift_mps), 3)


def rank_vessels_for_case(case_id, case_dir, spill_json_path):
    """
    Ranks AIS candidate vessels for a single case using 5 normalized attribution criteria.
    """
    # 1. Load spill.json
    with open(spill_json_path, "r") as f:
        spill_data = json.load(f)

    spill_area_km2 = spill_data.get("area_km2", 0.0)

    # 2. Load stage5/hindcast.json
    hindcast_path = os.path.join(case_dir, "stage5", "hindcast.json")
    if os.path.exists(hindcast_path):
        with open(hindcast_path, "r") as f:
            hindcast_data = json.load(f)

        det_pt = hindcast_data.get("detection_point", {})
        det_time_str = hindcast_data.get("detection_time", "2026-07-10T00:48:00Z")
        best_est = hindcast_data.get("best_estimate", {})
        origin_pt = best_est.get("origin", {})
        dur_hours = best_est.get("duration_hours", 48)
    else:
        det_pt = {"lat": spill_data.get("centroid_lat"), "lon": spill_data.get("centroid_lon")}
        det_time_str = "2026-07-10T00:48:00Z"
        origin_pt = det_pt
        dur_hours = 48

    # Parse ISO timestamps
    det_dt = datetime.fromisoformat(det_time_str.replace("Z", "+00:00"))
    rel_dt = det_dt - timedelta(hours=dur_hours) if 'timedelta' in globals() else datetime.fromtimestamp(det_dt.timestamp() - dur_hours * 3600, tz=timezone.utc)
    rel_time_str = rel_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    hindcast_speed_mps = compute_hindcast_drift_speed(det_pt, origin_pt, dur_hours)

    # 3. Load candidate vessels from stage6/ais_filtered.json or stage6/ais_all.json
    ais_path = os.path.join(case_dir, "stage6", "ais_filtered.json")
    if not os.path.exists(ais_path):
        ais_path = os.path.join(case_dir, "stage6", "ais_all.json")

    if not os.path.exists(ais_path):
        print(f"Warning: No Stage 6 AIS data found at '{ais_path}' for case '{case_id}'.", file=sys.stderr, flush=True)
        return None

    with open(ais_path, "r") as f:
        ais_data = json.load(f)

    vessels = ais_data.get("vessels", [])
    if not vessels:
        # Fallback to ais_all.json if ais_filtered.json is empty
        all_ais_path = os.path.join(case_dir, "stage6", "ais_all.json")
        if os.path.exists(all_ais_path):
            with open(all_ais_path, "r") as f:
                vessels = json.load(f).get("vessels", [])

    if not vessels:
        print(f"Warning: Candidate vessel list is empty for case '{case_id}'.", file=sys.stderr, flush=True)
        return None

    # 4. Compute 5 normalized criteria for each candidate vessel
    ranked_candidates = []

    for v in vessels:
        mmsi = v.get("mmsi") or v.get("vessel_id")
        name = v.get("name", f"VESSEL_{mmsi}")
        ship_type = v.get("ship_type", "Unknown")

        closest_dist_km = v.get("closest_distance_km", 0.0)

        # Time at closest approach relative to release_time_estimate
        time_at_closest_str = v.get("time_at_closest", rel_time_str)
        try:
            closest_dt = datetime.fromisoformat(time_at_closest_str.replace("Z", "+00:00"))
        except Exception:
            closest_dt = rel_dt

        time_offset_sec = (closest_dt - rel_dt).total_seconds()
        time_offset_hours = time_offset_sec / 3600.0
        abs_time_diff_hours = abs(time_offset_hours)

        # Vessel SOG in m/s
        avg_sog_kn = v.get("avg_sog_near_origin")
        if avg_sog_kn is None:
            # fallback to track average
            track = v.get("track", [])
            sogs = [pt["sog_kn"] for pt in track if "sog_kn" in pt]
            avg_sog_kn = float(np.mean(sogs)) if sogs else 5.0

        vessel_speed_mps = avg_sog_kn * 0.514444

        # Dwell minutes near origin
        dwell_minutes = v.get("dwell_minutes_near_origin", 0.0)

        # ── 5 NORMALISED CRITERIA (0 to 1) ───────────────────────────────────
        # 1. Proximity: 1 - (min_distance / 50.0) [50 km scale]
        prox_score = max(0.0, min(1.0, 1.0 - (closest_dist_km / 50.0)))

        # 2. Time: exp(-|timestamp_hours - spill_time_hours| / 6.0) [6 h scale]
        time_score = math.exp(-abs_time_diff_hours / 6.0)

        # 3. Speed: 1 - (abs(speed_mps - hindcast_drift_speed_mps) / 5.0)
        speed_diff_mps = abs(vessel_speed_mps - hindcast_speed_mps)
        speed_score = max(0.0, min(1.0, 1.0 - (speed_diff_mps / 5.0)))

        # 4. Dwell: exp(-abs(minutes_before_after) / 60.0)
        minutes_before_after = abs(time_offset_hours * 60.0)
        dwell_score = math.exp(-minutes_before_after / 60.0)

        # 5. Type: 1 if tanker, 0.8 if cargo, 0.6 if other
        st_lower = ship_type.lower()
        if "tanker" in st_lower:
            type_score = 1.0
            type_short = "tanker"
        elif "cargo" in st_lower:
            type_score = 0.8
            type_short = "cargo"
        else:
            type_score = 0.6
            type_short = "other"

        # ── COMPOSITE ATTRIBUTION SCORE ──────────────────────────────────────
        # Final score = 0.30*Proximity + 0.25*Time + 0.20*Speed + 0.15*Dwell + 0.10*Type
        final_score = (
            0.30 * prox_score +
            0.25 * time_score +
            0.20 * speed_score +
            0.15 * dwell_score +
            0.10 * type_score
        )
        final_score = round(float(final_score), 4)

        # ── JUSTIFICATION STRING FORMAT ──────────────────────────────────────
        # "Proximity 0.8 km at t-4.2 h | Speed match 1.2 m/s | Dwell 27 min before | Type = tanker"
        timing_rel = "before" if time_offset_hours <= 0 else "after"
        t_sign_str = f"{time_offset_hours:+.1f}"
        justification = (
            f"Proximity {closest_dist_km:.1f} km at t{t_sign_str} h | "
            f"Speed match {speed_diff_mps:.1f} m/s | "
            f"Dwell {abs(minutes_before_after):.0f} min {timing_rel} | "
            f"Type = {type_short}"
        )

        ranked_candidates.append({
            "mmsi": mmsi,
            "name": name,
            "ship_type": ship_type,
            "final_score": final_score,
            "proximity_score": round(float(prox_score), 4),
            "time_score": round(float(time_score), 4),
            "speed_score": round(float(speed_score), 4),
            "dwell_score": round(float(dwell_score), 4),
            "type_score": round(float(type_score), 4),
            "metrics": {
                "closest_distance_km": round(float(closest_dist_km), 2),
                "time_offset_hours": round(float(time_offset_hours), 2),
                "vessel_speed_mps": round(float(vessel_speed_mps), 2),
                "hindcast_speed_mps": round(float(hindcast_speed_mps), 2),
                "dwell_minutes": round(float(dwell_minutes), 1)
            },
            "justification": justification
        })

    # Sort candidates by final_score DESCENDING (Rank 1 = Highest score / primary suspect)
    ranked_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # Assign 1-based ranks
    for idx, c in enumerate(ranked_candidates):
        c["rank"] = idx + 1

    # 5. Output stage7/ranked_suspects.json
    stage7_dir = os.path.join(case_dir, "stage7")
    os.makedirs(stage7_dir, exist_ok=True)

    json_output = {
        "case_id": case_id,
        "data_source": "synthetic_demo_ais",
        "spill_area_km2": round(float(spill_area_km2), 4),
        "estimated_origin": origin_pt,
        "release_time_estimate": rel_time_str,
        "detection_time": det_time_str,
        "ranked_suspects": ranked_candidates
    }

    output_path = os.path.join(stage7_dir, "ranked_suspects.json")
    with open(output_path, "w") as f:
        json.dump(json_output, f, indent=2)

    return json_output


def main():
    print("=" * 85, flush=True)
    print(" SIH26143 — STAGE 7: SUSPECT VESSEL RANKING", flush=True)
    print("=" * 85, flush=True)

    case_dirs = find_case_folders()
    if not case_dirs:
        print("Error: No case folders found under 'cases/' or 'CASES/'.", flush=True)
        sys.exit(1)

    print(f"Found {len(case_dirs)} case folder(s) to process:", flush=True)
    for cid, cdir, sjson in case_dirs:
        print(f"  - [{cid}] in '{cdir}' using '{os.path.basename(sjson)}'", flush=True)
    print("-" * 85, flush=True)

    summary_results = []

    for case_id, case_dir, spill_json_path in case_dirs:
        try:
            res = rank_vessels_for_case(case_id, case_dir, spill_json_path)
            if res:
                summary_results.append(res)
        except Exception as e:
            print(f"Error ranking candidates for case '{case_id}': {e}", file=sys.stderr, flush=True)

    # Save combined stage7_summary.json
    cases_root = "cases" if os.path.exists("cases") else "CASES"
    summary_path = os.path.join(cases_root, "stage7_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    # 6. Print Console Table
    print("\n" + "=" * 95, flush=True)
    print(" STAGE 7 SUSPECT VESSEL ATTRIBUTION RANKING TABLE", flush=True)
    print("=" * 95, flush=True)

    for case_res in summary_results:
        cid = case_res["case_id"]
        rel_t = case_res.get("release_time_estimate", "N/A")
        area = case_res.get("spill_area_km2", 0.0)
        suspects = case_res.get("ranked_suspects", [])

        print(f"\nCASE: {cid} | Est. Release: {rel_t} | Area: {area:.2f} km²", flush=True)
        print("-" * 95, flush=True)
        print(f"{'Rank':<5} | {'MMSI':<10} | {'Vessel Name':<16} | {'Ship Type':<14} | {'Score':<6} | {'Justification'}", flush=True)
        print("-" * 95, flush=True)

        for s in suspects[:3]:  # Top 3 suspects
            rk = s["rank"]
            mmsi = s["mmsi"]
            name = s["name"][:15]
            stype = s["ship_type"][:13]
            score = f"{s['final_score']:.4f}"
            just = s["justification"]
            print(f"{rk:<5} | {mmsi:<10} | {name:<16} | {stype:<14} | {score:<6} | {just}", flush=True)
        print("-" * 95, flush=True)

    print(f"\nSummary JSON saved to: {summary_path}", flush=True)
    print("=" * 95, flush=True)

    print("\nStage 7 complete: Final suspect vessel attribution ranking generated.", flush=True)


if __name__ == "__main__":
    main()
