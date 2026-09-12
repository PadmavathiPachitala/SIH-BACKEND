# stage3b_age.py
"""Stage 3b – Age Estimation

This script reads spill characterization data (``spill.json``) and, optionally, a wind history NetCDF and a binary mask image to
estimate the oil‑slick age range.  The algorithm follows the specification provided in the
implementation plan:

1. **Compactness** – ``C = 4·π·A / P²`` when ``perimeter_km`` is present.
2. **Slick orientation** – Use ``orientation_deg`` from ``spill.json`` if available, otherwise compute the
   principal axis of the supplied mask via PCA.
3. **Primary estimate (wind history)** – Find the longest contiguous window ending at the detection time
   where the wind direction aligns with the slick orientation within a configurable tolerance.
4. **Secondary estimate (Fay sanity)** – For a configurable range of surface wind speeds ``V`` invert a
   gravity‑viscous spreading law to obtain feasible ages and derive a percentile band.
5. **Merge** – Intersect the wind‑derived age with the Fay band (when wind is available) and assign a confidence
   level.
6. **Output** – Write ``age_estimate.json`` that always contains a range (never a single‑point value) together
   with median age, confidence, compactness, orientation and honesty notes.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Optional imports – we guard them because the script must run even if the optional
# dependencies are not installed.  If they are missing, the corresponding feature
# (wind or mask) will be disabled.
try:
    import netCDF4  # type: ignore
except Exception:  # pragma: no cover
    netCDF4 = None
try:
    import imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_spill(spill_path: str) -> dict:
    """Load ``spill.json`` and return the parsed dictionary."""
    with open(spill_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_compactness(area_km2: float, perimeter_km: Optional[float]) -> Optional[float]:
    """Return compactness ``C = 4·π·A / P²`` if ``perimeter_km`` is given.

    The function returns ``None`` when the perimeter is missing.
    """
    if perimeter_km is None or perimeter_km == 0:
        return None
    return 4 * math.pi * area_km2 / (perimeter_km ** 2)


def orientation_from_mask(mask_path: str) -> Optional[float]:
    """Compute the principal orientation of a binary mask using PCA.

    The mask is expected to be a 2‑D image where non‑zero pixels belong to the slick.
    The orientation is returned in **degrees** measured clockwise from north (the same
    convention used for ``orientation_deg`` in ``spill.json``).
    """
    if imageio is None:
        return None
    try:
        img = imageio.imread(mask_path)
    except Exception:
        return None
    # Ensure we are working with a binary mask (any non‑zero is considered "slick")
    coords = np.column_stack(np.where(img != 0))
    if coords.shape[0] < 2:
        return None
    # Center the coordinates
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    # Covariance and eigen‑decomposition (PCA)
    cov = np.cov(centered, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    # Principal eigenvector (largest eigenvalue)
    principal = eig_vecs[:, np.argmax(eig_vals)]
    # Angle of the vector w.r.t. the x‑axis (east).  We need clockwise from north:
    angle_rad = math.atan2(principal[0], principal[1])  # swap because y is north
    angle_deg = math.degrees(angle_rad)
    # Convert to range [0, 360)
    angle_deg = angle_deg % 360
    return angle_deg


def angular_difference(a: float, b: float) -> float:
    """Smallest absolute angular difference between two headings (degrees)."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def load_wind_series(nc_path: str, var_u: str = "u10", var_v: str = "v10") -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
    """Load wind ``u`` and ``v`` components and the associated timestamps.

    Returns ``(u, v, timestamps)`` where ``u`` and ``v`` are 1‑D numpy arrays of the same
    length and ``timestamps`` is a list of ``datetime`` objects.
    """
    if netCDF4 is None:
        raise RuntimeError("netCDF4 is not available – wind processing cannot be performed.")
    ds = netCDF4.Dataset(nc_path, mode="r")
    u = ds.variables[var_u][:]
    v = ds.variables[var_v][:]
    # Attempt to read a time variable.  Many datasets expose ``time`` as hours since a
    # reference date; we handle the most common case (units like "hours since 1970-01-01 00:00:00").
    if "time" in ds.variables:
        time_var = ds.variables["time"]
        units = getattr(time_var, "units", "hours since 1970-01-01 00:00:00")
        try:
            base = datetime.strptime(units.split("since")[1].strip(), "%Y-%m-%d %H:%M:%S")
        except Exception:
            base = datetime(1970, 1, 1)
        timestamps = [base + timedelta(hours=float(t)) for t in time_var[:]]
    else:
        timestamps = []
    ds.close()
    return np.squeeze(u), np.squeeze(v), timestamps


def wind_direction_series(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Convert ``u``/``v`` wind components to headings in degrees (0‑360, clockwise from north)."""
    rad = np.arctan2(u, v)  # swapped order to get heading from north
    deg = np.degrees(rad) % 360
    return deg


def longest_consistent_window(
    wind_dirs: np.ndarray,
    detection_idx: int,
    orientation_deg: float,
    tolerance_deg: float = 25.0,
) -> int:
    """Return the length (in time steps) of the longest contiguous window ending at
    ``detection_idx`` where the absolute angular difference is below ``tolerance_deg``.
    """
    if detection_idx >= len(wind_dirs):
        detection_idx = len(wind_dirs) - 1
    length = 0
    for i in range(detection_idx, -1, -1):
        if angular_difference(wind_dirs[i], orientation_deg) <= tolerance_deg:
            length += 1
        else:
            break
    return length


def fay_age_range(
    observed_radius_km: float,
    v_min: float = 0.1,
    v_max: float = 5.0,
    v_step: float = 0.1,
    k_constant: float = 1.0,
) -> Tuple[float, float]:
    """Invert a simplified gravity‑viscous spreading law to obtain a feasible age band.

    The law (placeholder) is ``R(t) = k·(V·t)^{1/4}`` where ``R`` is the slick radius (km),
    ``V`` the surface wind speed (m s⁻¹) and ``t`` the age in **hours**.  Solving for ``t`` gives::

        t = (R/k)⁴ / V

    For each admissible wind speed ``V`` we compute a candidate age.  The 10‑th and 90‑th
    percentiles of the resulting distribution form the Fay age range.
    """
    vs = np.arange(v_min, v_max + v_step, v_step)
    # Convert V from m/s to km/h (1 m/s = 3.6 km/h) to keep units consistent with ``t`` in hours.
    vs_kmh = vs * 3.6
    ages = (observed_radius_km / k_constant) ** 4 / vs_kmh
    lower = np.percentile(ages, 10)
    upper = np.percentile(ages, 90)
    return float(lower), float(upper)


def intersect_ranges(a: Tuple[float, float], b: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """Return the intersection of two inclusive ranges or ``None`` if they do not overlap."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if lo <= hi:
        return lo, hi
    return None


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3b – Oil‑slick age estimation")
    parser.add_argument("--spill", required=True, help="Path to spill.json")
    parser.add_argument("--wind", help="Path to wind history NetCDF (optional)")
    parser.add_argument("--mask", help="Path to binary mask image for PCA orientation (optional)")
    parser.add_argument(
        "--config",
        help="Path to optional JSON config overriding defaults (tolerance, Vmin, Vmax, ...)",
    )
    parser.add_argument(
        "--output",
        default="age_estimate.json",
        help="Filename for the output JSON (written to the same folder as spill.json)",
    )
    args = parser.parse_args()

    cfg = {}
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not read config file: {e}. Using defaults.")

    tolerance_deg: float = cfg.get("tolerance_deg", 25.0)
    v_min: float = cfg.get("v_min", 0.1)
    v_max: float = cfg.get("v_max", 5.0)
    v_step: float = cfg.get("v_step", 0.1)
    k_constant: float = cfg.get("k_constant", 1.0)

    spill = load_spill(args.spill)
    area_km2 = spill.get("area_km2")
    perimeter_km = spill.get("perimeter_km")
    compactness = compute_compactness(area_km2, perimeter_km)

    orientation_deg: Optional[float] = spill.get("orientation_deg")
    if orientation_deg is None and args.mask:
        orientation_deg = orientation_from_mask(args.mask)
    if orientation_deg is None:
        orientation_deg = 0.0
        orientation_note = "orientation not provided and mask unavailable - assumed 0 deg"
    else:
        orientation_note = None

    wind_available = False
    age_wind_hours: Optional[int] = None
    notes: List[str] = []

    if args.wind:
        try:
            u, v, timestamps = load_wind_series(args.wind)
            wind_dirs = wind_direction_series(u, v)
            detection_time_str = spill.get("detection_time")
            if detection_time_str:
                detection_dt = datetime.fromisoformat(detection_time_str)
                if timestamps:
                    time_diffs = [abs((t - detection_dt).total_seconds()) for t in timestamps]
                    detection_idx = int(np.argmin(time_diffs))
                else:
                    detection_idx = len(wind_dirs) - 1
            else:
                detection_idx = len(wind_dirs) - 1

            window_len = longest_consistent_window(
                wind_dirs, detection_idx, float(orientation_deg), tolerance_deg
            )
            age_wind_hours = max(3, min(72, window_len)) if window_len > 0 else None
            wind_available = True
        except Exception as e:
            notes.append(f"[WARNING] Wind processing failed: {e}")
            wind_available = False
    else:
        notes.append("Wind data not supplied.")

    observed_radius_km = math.sqrt(area_km2 / math.pi)
    fay_lower, fay_upper = fay_age_range(
        observed_radius_km, v_min=v_min, v_max=v_max, v_step=v_step, k_constant=k_constant
    )
    # Clamp bounds within the realistic hindcast search window [3.0, 72.0]
    fay_lower = max(3.0, min(72.0, fay_lower))
    fay_upper = max(3.0, min(72.0, fay_upper))
    if fay_lower > fay_upper:
        fay_lower, fay_upper = fay_upper, fay_lower
    if abs(fay_upper - fay_lower) < 1.0:
        if fay_upper >= 72.0:
            fay_lower = max(3.0, 72.0 - 6.0)
        else:
            fay_upper = min(72.0, fay_lower + 6.0)

    if wind_available and age_wind_hours is not None:
        wind_range = (float(age_wind_hours), float(age_wind_hours))
        intersect = intersect_ranges(wind_range, (fay_lower, fay_upper))
        if intersect:
            final_lower, final_upper = intersect
            confidence = "high"
            notes.append("Wind steady and Fay sanity overlap - high confidence.")
        else:
            final_lower, final_upper = fay_lower, fay_upper
            confidence = "medium"
            notes.append("Wind steady but does not overlap Fay band - medium confidence.")
    else:
        final_lower, final_upper = fay_lower, fay_upper
        confidence = "low"
        notes.append("Wind data unavailable - confidence reduced.")

    # Guarantee monotonic range within [3.0, 72.0]
    final_lower = max(3.0, min(72.0, final_lower))
    final_upper = max(3.0, min(72.0, final_upper))
    if final_lower > final_upper:
        final_lower, final_upper = final_upper, final_lower
    if abs(final_upper - final_lower) < 1.0:
        if final_upper >= 72.0:
            final_lower = max(3.0, 72.0 - 1.0)
        else:
            final_upper = min(72.0, final_lower + 1.0)
        notes.append("Adjusted zero-width range to a minimal 1h span.")

    median_age = (final_lower + final_upper) / 2.0

    output = {
        "age_hours_range": [round(final_lower, 2), round(final_upper, 2)],
        "median_age_hours": round(median_age, 2),
        "confidence": confidence,
        "compactness": round(compactness, 4) if compactness is not None else None,
        "slick_orientation_deg": round(float(orientation_deg), 2),
        "notes": " ".join(notes),
    }

    output_path = os.path.join(os.path.dirname(args.spill), args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("--- Age Estimation Summary ---")
    print(f"Compactness: {compactness:.4f}" if compactness is not None else "Compactness: N/A")
    print(f"Slick orientation: {orientation_deg:.2f} deg")
    if wind_available:
        print(f"Wind-derived age (steady window): {age_wind_hours} h")
    else:
        print("Wind-derived age: unavailable")
    print(f"Fay-sanity age band: {fay_lower:.2f}-{fay_upper:.2f} h")
    print(f"Final age range: {final_lower:.2f}-{final_upper:.2f} h (median {median_age:.2f} h)")
    print(f"Confidence level: {confidence}")
    print("Notes:")
    for n in notes:
        print(f"  - {n}")



if __name__ == "__main__":
    main()
