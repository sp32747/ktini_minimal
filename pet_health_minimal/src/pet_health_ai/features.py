from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_generation import SUPERVISED_TASKS

RAW_SENSOR_COLUMNS = ["heart_rate", "spo2", "temperature", "light", "gps_lat", "gps_lon"]
SEQUENCE_FEATURES = [
    "heart_rate", "spo2", "temperature", "log_light", "gps_speed_mps",
    "hr_delta_from_baseline", "temp_delta_from_baseline", "spo2_delta_from_baseline",
    "hour_sin", "hour_cos",
]


@dataclass
class WindowedDataset:
    features: pd.DataFrame
    sequences: np.ndarray
    labels: np.ndarray
    task_labels: np.ndarray
    task_names: list[str]
    pet_ids: np.ndarray
    window_end_times: np.ndarray
    sequence_feature_names: list[str]


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1 - a, 1e-12)))


def clean_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {"pet_id", "timestamp", *RAW_SENSOR_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["pet_id", "timestamp"]).sort_values(["pet_id", "timestamp"]).reset_index(drop=True)
    valid_ranges = {
        "heart_rate": (30.0, 300.0), "spo2": (70.0, 100.0), "temperature": (35.0, 43.0),
        "light": (0.0, 200_000.0), "gps_lat": (-90.0, 90.0), "gps_lon": (-180.0, 180.0),
    }
    for col, (lo, hi) in valid_ranges.items():
        out.loc[~out[col].between(lo, hi), col] = np.nan
    for col in RAW_SENSOR_COLUMNS:
        out[col] = out.groupby("pet_id", group_keys=False)[col].apply(
            lambda s: s.interpolate(limit_direction="both").ffill().bfill()
        )
    if out[RAW_SENSOR_COLUMNS].isna().any().any():
        out[RAW_SENSOR_COLUMNS] = out[RAW_SENSOR_COLUMNS].fillna(out[RAW_SENSOR_COLUMNS].median(numeric_only=True))
    return out


def add_point_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["pet_id", "timestamp"]).reset_index(drop=True)
    out["gps_speed_mps"] = 0.0
    for _, idx in out.groupby("pet_id", sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        g = out.loc[idx_arr]
        lat, lon = g["gps_lat"].to_numpy(float), g["gps_lon"].to_numpy(float)
        times = g["timestamp"].astype("int64").to_numpy() / 1e9
        speed = np.zeros(len(g))
        if len(g) > 1:
            d = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
            speed[1:] = d / np.maximum(times[1:] - times[:-1], 1.0)
        out.loc[idx_arr, "gps_speed_mps"] = np.clip(speed, 0.0, 15.0)

    hour = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["log_light"] = np.log1p(np.clip(out["light"].to_numpy(float), 0, None))
    if "baseline_hr" not in out:
        out["baseline_hr"] = out.groupby("pet_id")["heart_rate"].transform("median")
    if "baseline_temp" not in out:
        out["baseline_temp"] = out.groupby("pet_id")["temperature"].transform("median")
    if "baseline_spo2" not in out:
        out["baseline_spo2"] = out.groupby("pet_id")["spo2"].transform("median")
    out["hr_delta_from_baseline"] = out["heart_rate"] - out["baseline_hr"]
    out["temp_delta_from_baseline"] = out["temperature"] - out["baseline_temp"]
    out["spo2_delta_from_baseline"] = out["spo2"] - out["baseline_spo2"]
    if "species" not in out: out["species"] = "unknown"
    if "age_years" not in out: out["age_years"] = 0.0
    if "weight_kg" not in out: out["weight_kg"] = 0.0
    out["species_code"] = out["species"].map({"dog": 0.0, "cat": 1.0}).fillna(2.0)
    return out


def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]): return 0.0
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8: return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def create_window_dataset(
    df: pd.DataFrame,
    window_minutes: int = 15,
    stride_minutes: int = 5,
    positive_window_fraction: float = 0.30,
    require_label: bool = True,
) -> WindowedDataset:
    working = df.copy()
    target_cols = [f"target_{t}" for t in SUPERVISED_TASKS]
    missing_targets = [c for c in target_cols if c not in working.columns]
    if missing_targets and require_label:
        # Backward-compatible derivation from risk_type where possible.
        if "risk_type" in working.columns:
            for task in SUPERVISED_TASKS:
                if f"target_{task}" not in working:
                    token = "resting_tachycardia" if task == "abnormal_resting_hr" else task
                    working[f"target_{task}"] = working["risk_type"].astype(str).str.contains(token).astype(int)
        else:
            raise ValueError(f"Training requires task target columns: {target_cols}")
    for c in target_cols:
        if c not in working: working[c] = 0
    if "label" not in working:
        working["label"] = working[target_cols].max(axis=1) if require_label else 0

    point_df = add_point_features(clean_sensor_data(working))
    feature_rows, seq_rows, labels, task_labels, pet_ids, end_times = [], [], [], [], [], []
    stats_cols = ["heart_rate", "spo2", "temperature", "light", "gps_speed_mps"]

    for pet_id, g in point_df.groupby("pet_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < window_minutes: continue
        for start in range(0, len(g) - window_minutes + 1, stride_minutes):
            w = g.iloc[start:start + window_minutes]
            row: dict[str, float] = {}
            for col in stats_cols:
                values = w[col].to_numpy(float)
                row[f"{col}_mean"] = float(np.mean(values)); row[f"{col}_std"] = float(np.std(values))
                row[f"{col}_min"] = float(np.min(values)); row[f"{col}_max"] = float(np.max(values))
                row[f"{col}_slope"] = _slope(values)
            row.update({
                "hr_delta_mean": float(w["hr_delta_from_baseline"].mean()),
                "hr_delta_max": float(w["hr_delta_from_baseline"].max()),
                "temp_delta_mean": float(w["temp_delta_from_baseline"].mean()),
                "temp_delta_max": float(w["temp_delta_from_baseline"].max()),
                "spo2_delta_mean": float(w["spo2_delta_from_baseline"].mean()),
                "spo2_delta_min": float(w["spo2_delta_from_baseline"].min()),
                "low_spo2_fraction": float((w["spo2"] < 94.0).mean()),
                "inactive_fraction": float((w["gps_speed_mps"] < 0.20).mean()),
                "active_fraction": float((w["gps_speed_mps"] > 0.70).mean()),
                "daylight_inactive_fraction": float(((w["gps_speed_mps"] < 0.20) & (w["light"] > 150)).mean()),
                "high_hr_while_inactive": float(((w["hr_delta_from_baseline"] > 30) & (w["gps_speed_mps"] < 0.20)).mean()),
                "hr_speed_corr": _safe_corr(w["heart_rate"].to_numpy(float), w["gps_speed_mps"].to_numpy(float)),
                "temperature_hr_corr": _safe_corr(w["temperature"].to_numpy(float), w["heart_rate"].to_numpy(float)),
                "hour_sin_mean": float(w["hour_sin"].mean()), "hour_cos_mean": float(w["hour_cos"].mean()),
                "age_years": float(w["age_years"].iloc[-1]), "weight_kg": float(w["weight_kg"].iloc[-1]),
                "species_code": float(w["species_code"].iloc[-1]), "baseline_hr": float(w["baseline_hr"].iloc[-1]),
                "baseline_temp": float(w["baseline_temp"].iloc[-1]), "baseline_spo2": float(w["baseline_spo2"].iloc[-1]),
            })
            task_y = [int(float(w[c].mean()) >= positive_window_fraction) for c in target_cols]
            feature_rows.append(row)
            seq_rows.append(w[SEQUENCE_FEATURES].to_numpy(np.float32))
            task_labels.append(task_y)
            labels.append(int(max(task_y)))
            pet_ids.append(str(pet_id)); end_times.append(w["timestamp"].iloc[-1].to_datetime64())

    if not feature_rows: raise ValueError("No windows were created. Check dataset size/window configuration.")
    feature_df = pd.DataFrame(feature_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return WindowedDataset(
        features=feature_df,
        sequences=np.stack(seq_rows).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        task_labels=np.asarray(task_labels, dtype=np.int64),
        task_names=list(SUPERVISED_TASKS),
        pet_ids=np.asarray(pet_ids),
        window_end_times=np.asarray(end_times),
        sequence_feature_names=list(SEQUENCE_FEATURES),
    )
