"""
run_pipeline.py — SIH26143 End-to-End Pipeline Orchestrator

Orchestrates all 8 pipeline stages for Arabian Sea Oil Spill Detection, Forecasting & Attribution:
- Stage 1: Deep SAR Semantic Segmentation (infer_geotiff.py)
- Stage 2: Post-Segmentation Swath Edge & Noise Filtering (filter_masks.py)
- Stage 3: Spill Feature Characterisation (stage3_characterise.py)
- Stage 3b: Oil Slick Age Estimation (stage3b_age.py)
- Stage 4/5: Ocean Current Hindcasting & Origin Estimation (stage5_hindcast.py)
- Stage 5b: Forward Trajectory & 96h Envelope Forecasting (stage5b_forecast.py)
- Stage 6: Demo AIS Track Generation & Real Filtering (stage6_ais_demo.py)
- Stage 7: Suspect Vessel Attribution Ranking (stage7_ranking.py)

Usage examples:
  python run_pipeline.py --case CASE_a1
  python run_pipeline.py --case CASE_a2
  python run_pipeline.py --case all
  python run_pipeline.py --case CASE_a1 --force
  python run_pipeline.py --case CASE_a1 --age-config path/to/config.json
  python run_pipeline.py --input path/to/spill.json [--age-config path/to/config.json]

Logs pipeline execution records to pipeline_runs/<case_id>_<timestamp>.json
"""

import os
import sys
import glob
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def resolve_case_dirs(case_arg):
    """
    Resolves target case directory paths under cases/ or CASES/,
    handling spaces vs underscores seamlessly (e.g. CASE_a1 -> CASE a1).
    """
    candidates_base = ["cases", "CASES"]
    found_cases = []
    seen = set()

    if case_arg.lower() == "all":
        for b in candidates_base:
            if not os.path.exists(b):
                continue
            for entry in sorted(os.listdir(b)):
                full = os.path.join(b, entry)
                if not os.path.isdir(full):
                    continue
                real_p = os.path.normcase(os.path.realpath(full))
                if real_p not in seen:
                    seen.add(real_p)
                    found_cases.append((entry, full))
        return found_cases

    # Match specific case argument (e.g. CASE_a1 -> CASE a1)
    target_norm = case_arg.lower().replace("_", " ").strip()

    for b in candidates_base:
        if not os.path.exists(b):
            continue
        for entry in os.listdir(b):
            full = os.path.join(b, entry)
            if not os.path.isdir(full):
                continue
            entry_norm = entry.lower().replace("_", " ").strip()
            if entry_norm == target_norm or entry.lower() == case_arg.lower():
                real_p = os.path.normcase(os.path.realpath(full))
                if real_p not in seen:
                    seen.add(real_p)
                    found_cases.append((entry, full))

    return found_cases


def run_stage_script(script_name, description, extra_args=None):
    """Executes a pipeline stage Python script via subprocess."""
    script_path = os.path.abspath(script_name)
    if not os.path.exists(script_path):
        print(f"  [FAIL] Script '{script_name}' not found at '{script_path}'", flush=True)
        return False

    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        if res.returncode == 0:
            return True
        else:
            print(f"  [FAIL] {description} failed with return code {res.returncode}", flush=True)
            if res.stderr and res.stderr.strip():
                print(f"  Error details (stderr):\n{res.stderr.strip()[:600]}", flush=True)
            if res.stdout and res.stdout.strip():
                print(f"  Error details (stdout):\n{res.stdout.strip()[-600:]}", flush=True)
            return False
    except Exception as e:
        print(f"  [FAIL] Error executing {script_name}: {e}", flush=True)
        return False


def run_pipeline_for_case(case_id, case_dir, force=False, age_config=None):
    """
    Runs full pipeline sequence for a single case directory.
    Skips existing stage outputs unless force=True.
    Stops early if filtering indicates no confident spill or empty mask.
    """
    print(f"\n" + "=" * 85, flush=True)
    print(f" RUNNING PIPELINE FOR CASE: [{case_id}]", flush=True)
    print(f" Path: {case_dir}", flush=True)
    print("=" * 85, flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_statuses = {}
    output_files = {}
    pipeline_status = "COMPLETED"

    # ── STAGE 3: SPILL CHARACTERISATION ─────────────────────────────────────
    spill_json_path = os.path.join(case_dir, "spill.json")
    if not os.path.exists(spill_json_path):
        spill_json_path_upper = os.path.join(case_dir, "SPILL.json")
        if os.path.exists(spill_json_path_upper):
            spill_json_path = spill_json_path_upper

    need_stage3 = force or not os.path.exists(spill_json_path)

    if need_stage3:
        print("\n[STAGE 3] Running Spill Characterisation...", flush=True)
        ok = run_stage_script("stage3_characterise.py", "Stage 3 Characterisation", extra_args=["--case", case_id])
        if not ok:
            stage_statuses["stage3_characterise"] = "FAIL"
            pipeline_status = "FAILED_STAGE3"
            return finalize_run_report(case_id, case_dir, timestamp, pipeline_status, stage_statuses, output_files)
        stage_statuses["stage3_characterise"] = "PASS"
    else:
        print("\n[STAGE 3] Output 'spill.json' exists. (Skipping, use --force to re-run)", flush=True)
        stage_statuses["stage3_characterise"] = "PASS (SKIPPED)"

    # Check Stage 3 status output
    if os.path.exists(spill_json_path):
        output_files["spill_json"] = spill_json_path.replace("\\", "/")
        with open(spill_json_path, "r") as f:
            spill_data = json.load(f)

        status_st3 = spill_data.get("status", "")
        if status_st3 == "empty_mask" or status_st3 == "no_confident_spill":
            print(f"\n[STOP] Pipeline stopped early: Stage 3 filter status is '{status_st3}'.", flush=True)
            pipeline_status = f"STOPPED_{status_st3.upper()}"
            return finalize_run_report(case_id, case_dir, timestamp, pipeline_status, stage_statuses, output_files)

        print(f"  -> Stage 3 Status : {status_st3} | Area: {spill_data.get('area_km2', 0):.2f} km² | Centroid: ({spill_data.get('centroid_lat')}, {spill_data.get('centroid_lon')})", flush=True)

    # ── STAGE 3b: OIL SPILL AGE ESTIMATION ──────────────────────────────────
    age_json_path = os.path.join(case_dir, "age.json")
    if not os.path.exists(age_json_path):
        age_alt = os.path.join(case_dir, "age_estimate.json")
        if os.path.exists(age_alt):
            age_json_path = age_alt

    need_stage3b = force or (age_config is not None) or (not os.path.exists(age_json_path))

    if need_stage3b and os.path.exists(spill_json_path):
        print("\n[STAGE 3b] Running Oil Spill Age Estimation...", flush=True)
        stage3b_args = ["--spill", spill_json_path, "--output", "age.json"]
        if age_config and os.path.exists(age_config):
            stage3b_args.extend(["--config", age_config])

        # Optional: detect mask if present in case directory
        mask_candidates = [
            os.path.join(case_dir, "filtered_mask.tif"),
            os.path.join(case_dir, "mask.tif"),
        ]
        mask_candidates.extend(glob.glob(os.path.join(case_dir, "*_filtered_mask.tif")))
        mask_candidates.extend(glob.glob(os.path.join(case_dir, "*mask*.png")))
        for m in mask_candidates:
            if os.path.exists(m):
                stage3b_args.extend(["--mask", m])
                break

        # Optional: detect wind NetCDF in Stage 4 data or case directory
        c_lower = case_id.lower().replace(" ", "_")
        c_dir_basename = os.path.basename(case_dir)
        search_patterns = [
            os.path.join("data", "stage4", c_lower, "*.nc"),
            os.path.join("data", "stage4", c_dir_basename, "*.nc"),
            os.path.join("data", "STAGE 4", c_dir_basename, "*.nc"),
            os.path.join("data", "STAGE 4", "RAW", "*.nc"),
            os.path.join("data", "STAGE 4", "*.nc"),
            os.path.join(case_dir, "*.nc"),
        ]
        wind_nc = None
        for pat in search_patterns:
            matches = glob.glob(pat)
            if matches:
                wind_nc = matches[0]
                break
        if wind_nc:
            stage3b_args.extend(["--wind", wind_nc])

        ok = run_stage_script("stage3b_age.py", "Stage 3b Age Estimation", extra_args=stage3b_args)
        if not ok:
            stage_statuses["stage3b_age"] = "FAIL"
        else:
            stage_statuses["stage3b_age"] = "PASS"
    elif os.path.exists(age_json_path):
        print("\n[STAGE 3b] Output 'age.json' exists. (Skipping, use --force to re-run)", flush=True)
        stage_statuses["stage3b_age"] = "PASS (SKIPPED)"

    # Ensure age.json is consistently available and mirrored to age_estimate.json
    final_age_file = os.path.join(case_dir, "age.json")
    if not os.path.exists(final_age_file):
        alt = os.path.join(case_dir, "age_estimate.json")
        if os.path.exists(alt):
            shutil.copy(alt, final_age_file)
            final_age_file = alt

    if os.path.exists(final_age_file):
        output_files["age_json"] = final_age_file.replace("\\", "/")
        try:
            with open(final_age_file, "r") as f:
                age_data = json.load(f)
            rng = age_data.get("age_hours_range", [])
            med = age_data.get("median_age_hours")
            cnf = age_data.get("confidence", "N/A")
            if rng and len(rng) == 2:
                print(f"  -> Stage 3b Status: PASS | Estimated Age: {rng[0]}-{rng[1]}h (Median: {med}h) | Conf: {cnf}", flush=True)
            else:
                print(f"  -> Stage 3b Status: PASS | Median Age: {med}h | Conf: {cnf}", flush=True)
        except Exception:
            pass

    # ── STAGE 4/5: HINDCASTING & ORIGIN ESTIMATION ───────────────────────────
    hindcast_json_path = os.path.join(case_dir, "stage5", "hindcast.json")
    need_stage5 = force or not os.path.exists(hindcast_json_path)

    if need_stage5:
        print("\n[STAGE 5] Running Ocean Current Hindcasting & Origin Estimation...", flush=True)
        ok = run_stage_script("stage5_hindcast.py", "Stage 5 Hindcasting")
        if not ok:
            stage_statuses["stage5_hindcast"] = "FAIL"
            pipeline_status = "FAILED_STAGE5"
            return finalize_run_report(case_id, case_dir, timestamp, pipeline_status, stage_statuses, output_files)
        stage_statuses["stage5_hindcast"] = "PASS"
    else:
        print("\n[STAGE 5] Output 'stage5/hindcast.json' exists. (Skipping, use --force to re-run)", flush=True)
        stage_statuses["stage5_hindcast"] = "PASS (SKIPPED)"

    if os.path.exists(hindcast_json_path):
        output_files["hindcast_json"] = hindcast_json_path.replace("\\", "/")
        with open(hindcast_json_path, "r") as f:
            hc_data = json.load(f)
        best_est = hc_data.get("best_estimate", {})
        ori = best_est.get("origin", {})
        print(f"  -> Stage 5 Status : PASS | Best Duration: {best_est.get('duration_hours')}h | Origin: ({ori.get('lat')}, {ori.get('lon')}) | Conf: {best_est.get('confidence')}", flush=True)

    # ── STAGE 5b: FORWARD TRAJECTORY & ENVELOPE FORECASTING ─────────────────
    forecast_geojson_path = os.path.join(case_dir, "forecast_envelope.geojson")
    need_stage5b = force or not os.path.exists(forecast_geojson_path)

    if need_stage5b:
        print("\n[STAGE 5b] Running Forward Trajectory & Envelope Forecasting (24h/72h/96h)...", flush=True)
        ok = run_stage_script("stage5b_forecast.py", "Stage 5b Forward Forecasting", extra_args=["--case", case_id])
        if not ok:
            stage_statuses["stage5b_forecast"] = "FAIL"
            pipeline_status = "FAILED_STAGE5b"
            return finalize_run_report(case_id, case_dir, timestamp, pipeline_status, stage_statuses, output_files)
        stage_statuses["stage5b_forecast"] = "PASS"
    else:
        print("\n[STAGE 5b] Output 'forecast_envelope.geojson' exists. (Skipping, use --force to re-run)", flush=True)
        stage_statuses["stage5b_forecast"] = "PASS (SKIPPED)"

    if os.path.exists(forecast_geojson_path):
        output_files["forecast_envelope"] = forecast_geojson_path.replace("\\", "/")
        forecast_json = os.path.join(case_dir, "forecast.json")
        if os.path.exists(forecast_json):
            output_files["forecast_json"] = forecast_json.replace("\\", "/")
            try:
                with open(forecast_json, "r") as f:
                    fc_data = json.load(f)
                src = fc_data.get("metocean_source", "NetCDF")
                h96 = fc_data.get("horizons", {}).get("96", fc_data.get("horizons", {}).get(96, {}))
                c96 = h96.get("centroid", {})
                print(f"  -> Stage 5b Status: PASS | Horizons: 24h, 72h, 96h | 96h Centroid: ({c96.get('lat')}, {c96.get('lon')}) | Envelopes: 50% & 90%", flush=True)
            except Exception:
                pass

    # ── STAGE 6: AIS CANDIDATE GENERATION & FILTERING ───────────────────────
    ais_filtered_json_path = os.path.join(case_dir, "stage6", "ais_filtered.json")
    need_stage6 = force or not os.path.exists(ais_filtered_json_path)

    if need_stage6:
        print("\n[STAGE 6] Running Demo AIS Track Generation & Real Filtering...", flush=True)
        ok = run_stage_script("stage6_ais_demo.py", "Stage 6 Demo AIS Filtering")
        if not ok:
            stage_statuses["stage6_ais"] = "FAIL"
            pipeline_status = "FAILED_STAGE6"
            return finalize_run_report(case_id, case_dir, timestamp, pipeline_status, stage_statuses, output_files)
        stage_statuses["stage6_ais"] = "PASS"
    else:
        print("\n[STAGE 6] Output 'stage6/ais_filtered.json' exists. (Skipping, use --force to re-run)", flush=True)
        stage_statuses["stage6_ais"] = "PASS (SKIPPED)"

    if os.path.exists(ais_filtered_json_path):
        output_files["ais_filtered_json"] = ais_filtered_json_path.replace("\\", "/")
        with open(ais_filtered_json_path, "r") as f:
            ais_data = json.load(f)
        n_passers = len(ais_data.get("vessels", []))
        print(f"  -> Stage 6 Status : PASS | Filtered Candidates: {n_passers} vessels", flush=True)

    # ── STAGE 7: SUSPECT VESSEL ATTRIBUTION RANKING ─────────────────────────
    ranking_json_path = os.path.join(case_dir, "stage7", "ranked_suspects.json")
    need_stage7 = force or not os.path.exists(ranking_json_path)

    if need_stage7:
        print("\n[STAGE 7] Running Suspect Vessel Attribution Ranking...", flush=True)
        ok = run_stage_script("stage7_ranking.py", "Stage 7 Suspect Ranking")
        if not ok:
            stage_statuses["stage7_ranking"] = "FAIL"
            pipeline_status = "FAILED_STAGE7"
            return finalize_run_report(case_id, case_dir, timestamp, pipeline_status, stage_statuses, output_files)
        stage_statuses["stage7_ranking"] = "PASS"
    else:
        print("\n[STAGE 7] Output 'stage7/ranked_suspects.json' exists. (Skipping, use --force to re-run)", flush=True)
        stage_statuses["stage7_ranking"] = "PASS (SKIPPED)"

    if os.path.exists(ranking_json_path):
        output_files["ranked_suspects_json"] = ranking_json_path.replace("\\", "/")
        with open(ranking_json_path, "r") as f:
            rk_data = json.load(f)
        suspects = rk_data.get("ranked_suspects", [])
        top_name = suspects[0]["name"] if suspects else "None"
        top_score = suspects[0]["final_score"] if suspects else 0.0
        print(f"  -> Stage 7 Status : PASS | Top Ranked Suspect: {top_name} (Attribution Score: {top_score:.4f})", flush=True)

    return finalize_run_report(case_id, case_dir, timestamp, pipeline_status, stage_statuses, output_files)


def finalize_run_report(case_id, case_dir, timestamp, pipeline_status, stage_statuses, output_files):
    """Saves pipeline execution record to pipeline_runs/<case_id>_<timestamp>.json."""
    os.makedirs("pipeline_runs", exist_ok=True)
    c_clean = case_id.replace(" ", "_")
    log_filename = f"{c_clean}_{timestamp}.json"
    log_path = os.path.join("pipeline_runs", log_filename)

    # Ensure pre-filtering stages are tracked in record
    if os.path.exists(os.path.join(case_dir, "filtered_mask.tif")) or glob.glob(os.path.join(case_dir, "*_filtered_mask.tif")):
        if "stage1_segmentation" not in stage_statuses:
            stage_statuses["stage1_segmentation"] = "PASS (COMPLETED)"
        if "stage2_filtering" not in stage_statuses:
            stage_statuses["stage2_filtering"] = "PASS (COMPLETED)"

    stage_order = [
        ("stage1_segmentation", "Deep SAR Segmentation (U-Net)"),
        ("stage2_filtering",    "Morphological & Swath Filter"),
        ("stage3_characterise",  "Spill Characterisation"),
        ("stage3b_age",          "Oil Slick Age Estimation"),
        ("stage5_hindcast",      "Lagrangian Hindcast (Origin)"),
        ("stage5b_forecast",     "Forward Forecast (24/72/96h)"),
        ("stage6_ais",           "AIS Candidate Track Gating"),
        ("stage7_ranking",       "MCDA Suspect Vessel Ranking")
    ]

    run_record = {
        "case_id": case_id,
        "timestamp": timestamp,
        "pipeline_status": pipeline_status,
        "stages": stage_statuses,
        "output_files": output_files
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)

    print("\n" + "=" * 85, flush=True)
    print(f" CASE [{case_id}] END-TO-END PIPELINE SUMMARY: {pipeline_status}", flush=True)
    print("=" * 85, flush=True)
    print(f"{'STAGE KEY':<22} | {'STAGE DESCRIPTION':<32} | {'STATUS':<20}", flush=True)
    print("-" * 85, flush=True)
    for stg_key, stg_name in stage_order:
        stg_stat = stage_statuses.get(stg_key, "SKIPPED / PENDING")
        print(f"{stg_key:<22} | {stg_name:<32} | {stg_stat:<20}", flush=True)
    print("-" * 85, flush=True)
    print(f" Pipeline run log saved to: {log_path}", flush=True)
    print("=" * 85, flush=True)

    return run_record



def main():
    parser = argparse.ArgumentParser(description="SIH26143 End-to-End Oil Spill Detection & Attribution Pipeline Orchestrator")
    parser.add_argument("--case", type=str, default="all", help="Target case ID (e.g. CASE_a1, CASE_a2, 'CASE a1', or 'all')")
    parser.add_argument("--geotiff", type=str, default=None, help="Optional direct path to a input SAR GeoTIFF")
    parser.add_argument("--case_id", type=str, default=None, help="Case ID to assign when running with --geotiff")
    parser.add_argument("--force", action="store_true", help="Force re-run all pipeline stages even if output files exist")
    parser.add_argument("--input", type=str, default=None, help="Direct path to spill.json to run Stage 3b age estimation")
    parser.add_argument("--age-config", type=str, default=None, help="Optional config JSON path for Stage 3b age estimation")
    args = parser.parse_args()

    print("=" * 85, flush=True)
    print(" SIH26143 — END-TO-END PIPELINE ORCHESTRATOR", flush=True)
    print("=" * 85, flush=True)

    # If a direct spill.json is provided, bypass case resolution and run age estimation directly
    if args.input:
        spill_path = Path(args.input)
        if not spill_path.is_file():
            print(f"[ERROR] Provided spill.json not found at {spill_path}", flush=True)
            sys.exit(1)

        base_dir = spill_path.parent
        print("\n[STAGE 3b] Running Age Estimation on provided spill.json ...", flush=True)
        age_cmd = [sys.executable, "stage3b_age.py", "--spill", str(spill_path), "--output", "age.json"]
        if args.age_config:
            age_cmd.extend(["--config", args.age_config])

        # Optional: detect mask and wind in base_dir
        mask_candidates = list(base_dir.glob("*mask*.tif")) + list(base_dir.glob("*mask*.png"))
        if mask_candidates:
            age_cmd.extend(["--mask", str(mask_candidates[0])])
        nc_candidates = list(base_dir.glob("*.nc"))
        if nc_candidates:
            age_cmd.extend(["--wind", str(nc_candidates[0])])

        try:
            res = subprocess.run(age_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            if res.returncode != 0:
                print(f"[ERROR] Age estimation failed with return code {res.returncode}", flush=True)
                if res.stderr:
                    print(f"Details: {res.stderr.strip()}", flush=True)
                sys.exit(res.returncode)
        except Exception as e:
            print(f"[ERROR] Age estimation failed: {e}", flush=True)
            sys.exit(1)

        # Ensure both age.json and age_estimate.json exist
        age_json_path = base_dir / "age.json"
        age_est_path = base_dir / "age_estimate.json"
        if not age_json_path.is_file() and age_est_path.is_file():
            shutil.copy(str(age_est_path), str(age_json_path))
        elif age_json_path.is_file() and not age_est_path.is_file():
            shutil.copy(str(age_json_path), str(age_est_path))

        print("\n=== Direct Run Summary ===")
        print(f"Spill file       : {spill_path}")
        print(f"Age output       : {age_json_path if age_json_path.is_file() else 'missing'}")
        if age_json_path.is_file():
            try:
                with open(age_json_path, "r", encoding="utf-8") as f:
                    age_info = json.load(f)
                rng = age_info.get("age_hours_range", [])
                med = age_info.get("median_age_hours")
                cnf = age_info.get("confidence")
                if rng and len(rng) == 2:
                    print(f"Estimated Age    : {rng[0]}-{rng[1]}h (Median: {med}h)")
                else:
                    print(f"Median Age       : {med}h")
                print(f"Confidence       : {cnf}")
            except Exception:
                pass
        sys.exit(0)

    # Normal case‑based pipeline execution
    target_case_arg = args.case_id if args.case_id else args.case
    case_matches = resolve_case_dirs(target_case_arg)

    if not case_matches:
        print(f"Error: No matching case directory found for '{target_case_arg}' under 'cases/' or 'CASES/'.", flush=True)
        sys.exit(1)

    print(f"Target Case(s) resolved ({len(case_matches)}):", flush=True)
    for cid, cpath in case_matches:
        print(f"  - [{cid}] in '{cpath}'", flush=True)

    pipeline_records = []
    for cid, cpath in case_matches:
        rec = run_pipeline_for_case(cid, cpath, force=args.force, age_config=args.age_config)
        pipeline_records.append(rec)

    print("\n" + "=" * 85, flush=True)
    print(" ALL PIPELINE RUNS COMPLETE", flush=True)
    print("=" * 85, flush=True)
    for r in pipeline_records:
        print(f" Case [{r['case_id']}] -> Status: {r['pipeline_status']}", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()

