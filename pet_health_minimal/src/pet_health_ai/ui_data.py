"""Input checks for interactive testing with the default one-minute models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import RAW_SENSOR_COLUMNS

REQUIRED_COLUMNS = ["pet_id", "timestamp", *RAW_SENSOR_COLUMNS]
SENSOR_RANGES = {
    "heart_rate": (30, 300), "spo2": (70, 100), "temperature": (35, 43),
    "light": (0, 200_000), "gps_lat": (-90, 90), "gps_lon": (-180, 180),
}


def prepare_sensor_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate before the batch pipeline interpolates any missing measurements."""
    missing = set(REQUIRED_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if raw.empty:
        raise ValueError("The CSV has no readings. Add at least 15 readings per pet.")
    frame = raw.copy()
    frame["pet_id"] = frame["pet_id"].astype("string").str.strip()
    if frame["pet_id"].isna().any() or frame["pet_id"].eq("").any():
        raise ValueError("Every reading needs a pet_id.")
    try:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        # Preserve local clock time: the models use hour-of-day features.
        if frame["timestamp"].dt.tz is not None:
            frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Use valid timestamps in one consistent local timezone.") from exc
    if frame["timestamp"].isna().any():
        raise ValueError("Every reading needs a valid timestamp.")
    if frame.duplicated(["pet_id", "timestamp"]).any():
        raise ValueError("Duplicate timestamps found for a pet. Keep one reading per minute.")
    frame = frame.sort_values(["pet_id", "timestamp"]).reset_index(drop=True)
    counts = frame.groupby("pet_id").size()
    short = counts[counts < 15]
    if len(short):
        raise ValueError("At least 15 readings are required for every pet. Too short: " + ", ".join(short.index))
    gaps = frame.groupby("pet_id")["timestamp"].diff().dropna()
    if not gaps.eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("Readings must be exactly one minute apart per pet. Resample to one-minute intervals and review gaps before uploading.")
    warnings = []
    invalid_count = 0
    for col, (lo, hi) in SENSOR_RANGES.items():
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        valid = frame[col].between(lo, hi)
        for pet, group in frame.groupby("pet_id"):
            if not valid.loc[group.index].any():
                raise ValueError(f"{pet}: {col} has no valid readings (expected {lo} to {hi}). Check units and sensor availability.")
        invalid_count += int((~valid).sum())
        frame.loc[~valid, col] = np.nan
    if invalid_count:
        warnings.append(f"{invalid_count:,} missing, nonnumeric, or out-of-range sensor values will be interpolated within each pet's history.")
    baselines = ["baseline_hr", "baseline_temp", "baseline_spo2"]
    for col in [*baselines, "age_years", "weight_kg"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            if not np.isfinite(frame[col].to_numpy(float)).all():
                raise ValueError(f"Optional column {col} must contain a number on every row, or be omitted entirely.")
    if any(col not in frame for col in baselines):
        warnings.append("Missing baselines will use each pet's medians from this file. Short or abnormal recordings can give misleading baselines.")
    return frame, warnings
