"""Shared, explicit feature preparation for livestock training and prediction."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import _haversine_m
from .livestock_data import LIVESTOCK_TASKS, temperature_humidity_index

FEATURE_VERSION = 1
RANGES = {
    "heart_rate": (30, 250), "spo2": (70, 100), "temperature": (35, 43),
    "light": (0, 200_000), "gps_lat": (-90, 90), "gps_lon": (-180, 180),
    "ambient_temperature_c": (-40, 60), "relative_humidity_pct": (0, 100),
    "wind_speed_mps": (0, 60),
}


@dataclass
class LivestockWindows:
    features: pd.DataFrame
    metadata: pd.DataFrame
    targets: np.ndarray | None


def create_livestock_windows(raw, window=15, stride=5, require_targets=False, positive_fraction=.30):
    if window < 3 or stride < 1 or not 0 < positive_fraction <= 1:
        raise ValueError("Invalid window, stride, or target fraction.")
    df = raw.copy()
    if "animal_id" not in df and "pet_id" in df:
        df["animal_id"] = df["pet_id"]
    required = {"animal_id", "timestamp", "species", *RANGES}
    target_cols = [f"target_{task}" for task in LIVESTOCK_TASKS]
    if require_targets:
        required.update(target_cols)
        required.add("region_id")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing livestock input columns: " + ", ".join(sorted(missing)))
    if df.empty:
        raise ValueError("No livestock readings supplied.")
    df["animal_id"] = df.animal_id.astype("string").str.strip()
    if df.animal_id.isna().any() or df.animal_id.eq("").any():
        raise ValueError("Each reading needs an animal ID.")
    df["species"] = df.species.astype("string").str.lower().str.strip()
    if not df.species.isin(["cow", "buffalo"]).all():
        raise ValueError("Livestock model supports only cow and buffalo species.")
    df["timestamp"] = pd.to_datetime(df.timestamp, errors="raise")
    if df.timestamp.isna().any():
        raise ValueError("Each reading needs a valid timestamp.")
    if df.timestamp.dt.tz is not None:
        df["timestamp"] = df.timestamp.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    if df.duplicated(["animal_id", "timestamp"]).any():
        raise ValueError("Duplicate timestamps for an animal.")
    if require_targets:
        for col in target_cols:
            df[col] = pd.to_numeric(df[col], errors="raise")
            if not df[col].isin([0, 1]).all():
                raise ValueError(f"{col} must contain only binary labels 0 or 1.")
    features, metadata, targets = [], [], []
    for animal, group in df.groupby("animal_id", sort=True):
        g = group.sort_values("timestamp").reset_index(drop=True)
        if g.species.nunique() != 1:
            raise ValueError(f"{animal}: species must be constant across the recording.")
        if len(g) < window:
            raise ValueError(f"{animal}: at least {window} readings are required.")
        if not g.timestamp.diff().dropna().eq(pd.Timedelta(minutes=1)).all():
            raise ValueError(f"{animal}: readings must be exactly one minute apart; resample and review gaps first.")
        species = g.species.iloc[0]
        defaults = {"baseline_hr": 64. if species == "cow" else 61.,
                    "baseline_temp": 38.5 if species == "cow" else 38.2, "baseline_spo2": 97.5,
                    "age_years": 5., "weight_kg": 450. if species == "cow" else 550.}
        for col, default in defaults.items():
            g[col] = default if col not in g else pd.to_numeric(g[col], errors="coerce")
            if not np.isfinite(g[col]).all():
                raise ValueError(f"{animal}: {col} must be finite, or omit the column to use species defaults.")
        fallback = {"heart_rate": g.baseline_hr.iloc[0], "spo2": g.baseline_spo2.iloc[0],
                    "temperature": g.baseline_temp.iloc[0], "light": 0.}
        for col, limits in RANGES.items():
            values = pd.to_numeric(g[col], errors="coerce")
            values = values.where(values.between(*limits))
            if not values.notna().any():
                raise ValueError(f"{animal}: {col} has no valid readings; check units.")
            # Only previous observations fill gaps. No interpolation from future values.
            values = values.ffill()
            if col in fallback:
                values = values.fillna(fallback[col])
            if values.isna().any():
                raise ValueError(f"{animal}: {col} needs a valid first reading.")
            g[col] = values
        lat, lon = g.gps_lat.to_numpy(), g.gps_lon.to_numpy()
        g["gps_speed_mps"] = np.r_[0, _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]) / 60].clip(0, 15)
        g["log_light"] = np.log1p(g.light)
        g["thi"] = temperature_humidity_index(g.ambient_temperature_c, g.relative_humidity_pct)
        g["hr_delta"] = g.heart_rate - g.baseline_hr
        g["temp_delta"] = g.temperature - g.baseline_temp
        g["spo2_delta"] = g.spo2 - g.baseline_spo2
        # Thirty-minute historical context helps represent sustained exposure.
        g["thi_history"] = g.thi.rolling(30, min_periods=1).mean()
        g["ambient_history"] = g.ambient_temperature_c.rolling(30, min_periods=1).mean()
        columns = ["heart_rate", "spo2", "temperature", "log_light", "gps_speed_mps",
                   "ambient_temperature_c", "relative_humidity_pct", "wind_speed_mps",
                   "thi", "hr_delta", "temp_delta", "spo2_delta", "thi_history", "ambient_history"]
        ends = np.arange(window - 1, len(g), stride)
        x = {}
        slope_axis = np.arange(window) - (window - 1) / 2
        for col in columns:
            sequences = np.lib.stride_tricks.sliding_window_view(g[col].to_numpy(float), window)[::stride]
            for name, values in [("mean", sequences.mean(axis=1)), ("std", sequences.std(axis=1)),
                                 ("min", sequences.min(axis=1)), ("max", sequences.max(axis=1)),
                                 ("slope", sequences @ slope_axis / (slope_axis @ slope_axis))]:
                x[f"{col}_{name}"] = values
        x["inactive_fraction"] = (g.gps_speed_mps < .2).rolling(window).mean().iloc[ends].to_numpy()
        x["high_hr_inactive_fraction"] = ((g.hr_delta > 25) & (g.gps_speed_mps < .2)).rolling(window).mean().iloc[ends].to_numpy()
        hour = g.timestamp.dt.hour + g.timestamp.dt.minute / 60
        x["hour_sin"] = np.sin(2 * np.pi * hour.iloc[ends].to_numpy() / 24)
        x["hour_cos"] = np.cos(2 * np.pi * hour.iloc[ends].to_numpy() / 24)
        x["species_cow"] = float(species == "cow")
        x["species_buffalo"] = float(species == "buffalo")
        for col in defaults:
            x[col] = g[col].iloc[ends].to_numpy(float)
        features.append(pd.DataFrame(x))
        region = g["region_id"].astype(str) if "region_id" in g else pd.Series("unknown", index=g.index)
        if region.nunique() != 1:
            raise ValueError(f"{animal}: region_id must be constant within a recording.")
        metadata.append(pd.DataFrame({"animal_id": animal, "species": species,
                                      "region_id": region.iloc[0], "window_end": g.timestamp.iloc[ends].to_numpy()}))
        if require_targets:
            targets.append((g[target_cols].rolling(window).mean().iloc[ends].to_numpy() >= positive_fraction).astype(int))
    result = LivestockWindows(pd.concat(features, ignore_index=True), pd.concat(metadata, ignore_index=True),
                              np.concatenate(targets) if require_targets else None)
    if not np.isfinite(result.features.to_numpy()).all():
        raise ValueError("Nonfinite livestock features after preparation.")
    return result
