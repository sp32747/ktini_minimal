"""Synthetic adult cows and buffaloes under five illustrative Indian climates.

These are engineering scenarios, not weather observations or veterinary rules.
See data/livestock/README.md for assumptions and literature references.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LIVESTOCK_TASKS = (
    "hypoxemia", "fever", "abnormal_resting_hr", "activity_reduction",
    "heat_stress", "cold_stress",
)


@dataclass(frozen=True)
class ClimateRegion:
    region_id: str
    location: str
    state: str
    climate: str
    start_date: str
    temperature_min_c: float
    temperature_max_c: float
    mean_humidity_pct: float
    mean_wind_mps: float
    latitude: float
    longitude: float


# Chosen simulation envelopes, not IMD normals or measurements for these dates.
REGIONS = (
    ClimateRegion("RJ", "Jaisalmer", "Rajasthan", "hot_arid", "2026-05-01", 24, 44, 25, 3.0, 26.9157, 70.9083),
    ClimateRegion("KL", "Thrissur", "Kerala", "hot_humid", "2026-07-01", 25, 34, 85, 1.4, 10.5276, 76.2144),
    ClimateRegion("PB", "Ludhiana", "Punjab", "warm_plains", "2026-05-01", 18, 37, 50, 2.0, 30.9010, 75.8573),
    ClimateRegion("KA", "Bengaluru", "Karnataka", "mild_plateau", "2026-02-01", 16, 29, 60, 2.1, 12.9716, 77.5946),
    ClimateRegion("JK", "Srinagar", "Jammu and Kashmir", "cold_winter", "2026-01-01", -6, 10, 70, 2.5, 34.0837, 74.7973),
)


def temperature_humidity_index(temperature_c, humidity_pct):
    """Buffington-style THI using Celsius and RH percent; not a cold index."""
    return 0.8 * temperature_c + (humidity_pct / 100) * (temperature_c - 14.4) + 46.4


def _smooth(values, span):
    return pd.Series(values).ewm(span=max(1, span), adjust=False).mean().to_numpy()


def _weather(region: ClimateRegion, hours: int, sample_minutes: int, seed: int):
    rng = np.random.default_rng(seed)
    n = hours * 60 // sample_minutes
    timestamps = pd.date_range(region.start_date, periods=n, freq=f"{sample_minutes}min")
    hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60
    days = np.arange(n) * sample_minutes // (24 * 60)
    daily_offset = rng.normal(0, .7, int(days.max()) + 1)[days]
    # Peak temperature at approximately 15:00; coolest at approximately 03:00.
    cycle = np.cos(2 * np.pi * (hour - 15) / 24)
    temp = ((region.temperature_min_c + region.temperature_max_c) / 2
            + (region.temperature_max_c - region.temperature_min_c) / 2 * cycle
            + daily_offset + _smooth(rng.normal(0, .4, n), 10 / sample_minutes))
    humidity = np.clip(region.mean_humidity_pct - 10 * cycle
                       + _smooth(rng.normal(0, 4, n), 15 / sample_minutes), 10, 99)
    wind = np.clip(region.mean_wind_mps + .6 * cycle
                   + _smooth(rng.normal(0, .5, n), 10 / sample_minutes), .1, 8)
    daylight = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    return timestamps, hour, temp, humidity, wind, daylight


def generate_livestock_data(
    cows: int = 15, buffaloes: int = 15, hours: int = 72,
    sample_minutes: int = 1, seed: int = 20260922,
    missing_fraction: float = .002, outlier_fraction: float = .0005,
) -> pd.DataFrame:
    """Allocate each species evenly across regions; preserve persistent animal IDs."""
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (cows, buffaloes, hours, sample_minutes)):
        raise ValueError("Animal counts, hours, and sample_minutes must be integers.")
    if cows < 5 or buffaloes < 5 or cows % 5 or buffaloes % 5:
        raise ValueError("Use at least 5 of each species, with counts divisible by 5 regions.")
    if hours < 24 or sample_minutes < 1 or sample_minutes > 5 or 60 % sample_minutes:
        raise ValueError("Use at least 24 hours and a sampling interval of 1, 2, 3, 4, or 5 minutes.")
    if not (0 <= missing_fraction <= .1 and 0 <= outlier_fraction <= .05):
        raise ValueError("Use missing_fraction in [0, .1] and outlier_fraction in [0, .05].")
    frames = []
    scenarios = ("reference", "fever", "hypoxemia", "abnormal_resting_hr", "activity_reduction", "fever+hypoxemia")
    animal_index = 0
    for r_index, region in enumerate(REGIONS):
        times, hour, ambient, humidity, wind, daylight = _weather(region, hours, sample_minutes, seed + r_index)
        n = len(times)
        minutes = np.arange(n) * sample_minutes
        thi = temperature_humidity_index(ambient, humidity)
        for species, count in (("cow", cows), ("buffalo", buffaloes)):
            for local in range(count // len(REGIONS)):
                animal_index += 1
                rng = np.random.default_rng(seed + 1000 + animal_index)
                animal_id = f"{species.upper()}_{region.region_id}_{local + 1:03d}"
                baseline_hr = float(np.clip(rng.normal(64 if species == "cow" else 61, 4), 50, 80))
                baseline_temp = float(rng.normal(38.5 if species == "cow" else 38.2, .15))
                baseline_spo2 = float(np.clip(rng.normal(97.5, .5), 95, 99))
                baseline_resp = float(rng.uniform(16, 23))
                age = float(rng.uniform(3, 9))
                weight = float(np.clip(rng.normal(450 if species == "cow" else 550, 55), 300, 750))
                shade = float(rng.uniform(.15, .85))
                cooling = float(rng.uniform(.05, .75))
                shelter = float(rng.uniform(.2, .85))
                sensitivity = float(rng.uniform(.85, 1.2) * (1.12 if species == "buffalo" else 1))
                # Individual engineering parameters, NOT diagnostic thresholds.
                heat_onset = float(rng.uniform(70, 75))
                heat_drive = np.clip((thi - heat_onset) / 15, 0, 1.5)
                heat = _smooth(heat_drive * sensitivity * (1 - .25 * shade - .3 * cooling), 30 / sample_minutes)
                cold_drive = np.clip((6 - ambient) / 15 + .03 * wind, 0, 1.5)
                cold_drive = np.where(ambient < 8, cold_drive, 0)
                cold = _smooth(cold_drive * (1 - .55 * shelter), 40 / sample_minutes)
                heat_target = (pd.Series(heat).rolling(max(1, 30 // sample_minutes), min_periods=1).mean().to_numpy() > .35).astype(np.int8)
                cold_target = (pd.Series(cold).rolling(max(1, 30 // sample_minutes), min_periods=1).mean().to_numpy() > .32).astype(np.int8)

                schedule = {task: np.zeros(n, dtype=np.int8) for task in LIVESTOCK_TASKS[:4]}
                scenario = scenarios[(r_index * (count // len(REGIONS)) + local
                                      + (3 if species == "buffalo" else 0)) % len(scenarios)]
                # A two-hour injected episode each day, during morning activity.
                for day in range((hours + 23) // 24):
                    start = day * 1440 + int(rng.integers(6 * 60, 8 * 60))
                    active = (minutes >= start) & (minutes < start + 120)
                    for task in scenario.split("+"):
                        if task in schedule:
                            schedule[task][active] = 1
                activity = np.clip(.06 + .65 * np.exp(-.5 * ((hour - 7.5) / 1.7) ** 2)
                                   + .6 * np.exp(-.5 * ((hour - 17.5) / 1.8) ** 2)
                                   + _smooth(rng.normal(0, .06, n), 4), .01, 1)
                activity *= np.clip(1 - .4 * heat - .35 * cold, .15, 1)
                activity *= np.where(schedule["activity_reduction"] | schedule["abnormal_resting_hr"], .10, 1)
                activity *= np.where(schedule["fever"], .6, 1)
                hr = baseline_hr + 27 * activity + 7 * heat + 4 * cold
                hr += 12 * schedule["fever"] + 8 * schedule["hypoxemia"] + 35 * schedule["abnormal_resting_hr"]
                hr += _smooth(rng.normal(0, 3, n), 3)
                body_temp = baseline_temp + .12 * activity + .7 * heat - .35 * cold
                body_temp += 1.1 * _smooth(schedule["fever"], 12 / sample_minutes) + _smooth(rng.normal(0, .06, n), 5)
                spo2 = baseline_spo2 - .2 * activity - 7 * schedule["hypoxemia"] + _smooth(rng.normal(0, .3, n), 3)
                respiration = baseline_resp + 10 * activity + 28 * heat + 5 * schedule["fever"]
                respiration += _smooth(rng.normal(0, 1.5, n), 4)
                # Bounded farm movement; observed speed is derived from these positions.
                bearing = rng.uniform(0, 2 * np.pi, n)
                distance = np.clip(.01 + .7 * activity + rng.normal(0, .01, n), 0, 1.2) * sample_minutes * 60
                east, north = np.zeros(n), np.zeros(n)
                for i in range(1, n):
                    east[i] = np.clip(east[i - 1] + distance[i] * np.cos(bearing[i]), -220, 220)
                    north[i] = np.clip(north[i - 1] + distance[i] * np.sin(bearing[i]), -220, 220)
                speed = np.r_[0, np.hypot(np.diff(east), np.diff(north)) / (sample_minutes * 60)]
                # Nearby hypothetical farms share regional weather, not real farm locations.
                home_lat = region.latitude + .03
                home_lon = region.longitude + .03
                latitude = home_lat + north / 111_320
                longitude = home_lon + east / (111_320 * np.cos(np.radians(home_lat)))
                schedule["heat_stress"], schedule["cold_stress"] = heat_target, cold_target
                matrix = np.column_stack([schedule[t] for t in LIVESTOCK_TASKS])
                risk = ["+".join(t for t, flag in zip(LIVESTOCK_TASKS, row) if flag) or "no_injected_risk" for row in matrix]
                frame = pd.DataFrame({
                    "animal_id": animal_id, "pet_id": animal_id, "timestamp": times,
                    "timezone": "Asia/Kolkata", "species": species, "sex": "female",
                    "life_stage": "adult", "region_id": region.region_id, "region": region.location,
                    "state": region.state, "climate_zone": region.climate,
                    "age_years": age, "weight_kg": weight,
                    "baseline_hr": baseline_hr, "baseline_spo2": baseline_spo2, "baseline_temp": baseline_temp,
                    "heart_rate": np.clip(hr, 35, 180), "spo2": np.clip(spo2, 75, 100),
                    "temperature": np.clip(body_temp, 35, 42.5),
                    "light": np.maximum(0, daylight * 55000 * (1 - .6 * shade) + rng.normal(0, 80, n)),
                    "gps_lat": latitude, "gps_lon": longitude, "gps_speed_mps": speed,
                    "ambient_temperature_c": ambient, "relative_humidity_pct": humidity,
                    "wind_speed_mps": wind, "temperature_humidity_index": thi,
                    "respiration_rate_bpm": np.clip(respiration, 8, 100),
                    "shade_access_fraction": shade, "cooling_access_fraction": cooling,
                    "shelter_protection_fraction": shelter,
                    "injected_scenario": scenario, "label": matrix.max(axis=1), "risk_type": risk,
                    **{f"target_{task}": schedule[task] for task in LIVESTOCK_TASKS},
                })
                # Missing/outlier flags are simulator metadata, never prediction features.
                frame["sensor_missing_count"] = 0
                frame["sensor_outlier_count"] = 0
                for col, bad_values in {"heart_rate": (10., 350.), "spo2": (45., 110.),
                                        "temperature": (30., 45.), "light": (-100., -50.)}.items():
                    ids = rng.permutation(n)
                    nm, no = int(n * missing_fraction), int(n * outlier_fraction)
                    missing_ids, outlier_ids = ids[:nm], ids[nm:nm + no]
                    frame.loc[missing_ids, col] = np.nan
                    frame.loc[outlier_ids, col] = rng.choice(bad_values, len(outlier_ids))
                    frame.loc[missing_ids, "sensor_missing_count"] += 1
                    frame.loc[outlier_ids, "sensor_outlier_count"] += 1
                frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["animal_id", "timestamp"]).reset_index(drop=True)


def save_livestock_data(output_dir: str | Path, **kwargs) -> pd.DataFrame:
    """Save labelled training data, unlabelled inputs, per-region files and metadata."""
    frame = generate_livestock_data(**kwargs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "synthetic_livestock_sensor_data.csv", index=False, float_format="%.6f")
    hidden = {"label", "risk_type", "injected_scenario", "sensor_missing_count", "sensor_outlier_count"}
    inputs = frame.drop(columns=[c for c in frame if c.startswith("target_") or c in hidden])
    inputs.to_csv(output / "livestock_inference_input.csv", index=False, float_format="%.6f")
    regions_dir = output / "regions"
    regions_dir.mkdir(exist_ok=True)
    for region_id, group in frame.groupby("region_id"):
        group.to_csv(regions_dir / f"{region_id}_livestock.csv", index=False, float_format="%.6f")
    summary = frame.groupby(["region_id", "region", "climate_zone", "species"], as_index=False).agg(
        animals=("animal_id", "nunique"), readings=("timestamp", "size"),
        ambient_min_c=("ambient_temperature_c", "min"), ambient_max_c=("ambient_temperature_c", "max"),
        mean_humidity_pct=("relative_humidity_pct", "mean"),
        heat_stress_fraction=("target_heat_stress", "mean"), cold_stress_fraction=("target_cold_stress", "mean"),
    )
    summary.to_csv(output / "region_species_summary.csv", index=False, float_format="%.4f")
    profile_columns = ["animal_id", "species", "region_id", "region", "state", "climate_zone", "age_years", "weight_kg", "baseline_hr", "baseline_temp", "baseline_spo2", "shade_access_fraction", "cooling_access_fraction", "shelter_protection_fraction", "injected_scenario"]
    frame[profile_columns].drop_duplicates("animal_id").to_csv(output / "animal_profiles.csv", index=False, float_format="%.6f")
    metadata = {
        "synthetic": True, "animals": int(frame.animal_id.nunique()), "rows": len(frame),
        "configuration_overrides": kwargs, "regions": [asdict(r) for r in REGIONS],
        "timezone": "Asia/Kolkata", "temperature_column": "synthetic body/core-like temperature, not ambient or skin temperature",
        "thi_formula": "0.8*T + RH/100*(T-14.4) + 46.4; T in Celsius, RH in percent",
        "targets": list(LIVESTOCK_TASKS),
        "limitations": [
            "Region envelopes and physiology coefficients are chosen simulation assumptions, not weather observations or validated veterinary thresholds.",
            "Different regional start dates represent contrasting seasonal scenarios, not simultaneous nationwide weather.",
            "Adult female cows and river-type buffaloes; breeds, pregnancy, lactation stage, feed and disease prevalence are not calibrated.",
            "Heat and cold targets are generated from synthetic exposure rules; they are not independently annotated outcomes.",
            "SpO2 and respiration are idealized signals; real hardware availability and measurement validity must be assessed.",
            "The existing dog/cat model feature mapping, tasks and health-score weights are not suitable livestock models.",
        ],
    }
    (output / "generation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame
