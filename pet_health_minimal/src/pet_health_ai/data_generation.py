from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SUPERVISED_TASKS = ["hypoxemia", "fever", "abnormal_resting_hr", "activity_reduction"]


@dataclass
class PetProfile:
    pet_id: str
    species: str
    age_years: float
    weight_kg: float
    baseline_hr: float
    baseline_spo2: float
    baseline_temp: float
    home_lat: float
    home_lon: float


def generate_pet_profiles(n_pets: int, seed: int = 42) -> list[PetProfile]:
    rng = np.random.default_rng(seed)
    profiles: list[PetProfile] = []
    for i in range(n_pets):
        species = "dog" if rng.random() < 0.68 else "cat"
        if species == "dog":
            weight = float(np.clip(rng.lognormal(mean=2.55, sigma=0.45), 3.0, 45.0))
            age = float(np.clip(rng.normal(6.0, 3.2), 0.5, 16.0))
            baseline_hr = float(np.clip(105 - 0.75 * weight + rng.normal(0, 7), 55, 125))
            baseline_temp = float(np.clip(rng.normal(38.55, 0.18), 38.0, 39.0))
        else:
            weight = float(np.clip(rng.normal(4.8, 1.2), 2.2, 8.5))
            age = float(np.clip(rng.normal(6.5, 3.6), 0.5, 18.0))
            baseline_hr = float(np.clip(rng.normal(155, 14), 115, 200))
            baseline_temp = float(np.clip(rng.normal(38.6, 0.18), 38.0, 39.1))
        baseline_spo2 = float(np.clip(rng.normal(97.8, 0.7), 95.5, 99.5))
        profiles.append(
            PetProfile(
                pet_id=f"P{i + 1:03d}",
                species=species,
                age_years=age,
                weight_kg=weight,
                baseline_hr=baseline_hr,
                baseline_spo2=baseline_spo2,
                baseline_temp=baseline_temp,
                home_lat=12.90 + rng.uniform(-0.08, 0.08),
                home_lon=77.60 + rng.uniform(-0.08, 0.08),
            )
        )
    return profiles


def _activity_signal(hour: float, rng: np.random.Generator) -> float:
    morning = math.exp(-0.5 * ((hour - 7.5) / 1.7) ** 2)
    evening = math.exp(-0.5 * ((hour - 18.5) / 2.0) ** 2)
    midday = 0.35 * math.exp(-0.5 * ((hour - 12.5) / 2.8) ** 2)
    return float(np.clip(0.62 * morning + 0.72 * evening + midday + rng.normal(0, 0.08), 0.0, 1.0))


def _light_from_hour(hour: float, rng: np.random.Generator) -> float:
    daylight = max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0))
    return float(max(0.0, 25.0 + 850.0 * daylight + rng.normal(0, 35.0)))


def _make_multitask_schedule(n_steps: int, rng: np.random.Generator, sample_minutes: int = 1) -> dict[str, np.ndarray]:
    """Create overlapping contiguous condition episodes.

    Each synthetic pet receives at least one episode for every supervised task when the
    history is long enough. This keeps pet-level train/validation/test splits usable in
    small smoke tests while still allowing overlaps and healthy periods.
    """
    targets = {name: np.zeros(n_steps, dtype=np.int8) for name in SUPERVISED_TASKS}
    min_duration = max(8, min(18, n_steps // 12))
    max_duration = max(min_duration + 1, min(70, max(min_duration + 2, n_steps // 5)))

    events = list(SUPERVISED_TASKS)
    extra = max(0, int(round(n_steps / (24 * 60) * rng.uniform(1.5, 3.5))))
    if extra:
        events.extend(rng.choice(SUPERVISED_TASKS, size=extra, replace=True).tolist())
    rng.shuffle(events)

    for task in events:
        duration = int(rng.integers(min_duration, max_duration + 1))
        duration = min(duration, max(1, n_steps - 1))
        if n_steps <= duration + 2:
            start = 0
        elif task == "activity_reduction":
            candidates = np.arange(0, n_steps - duration, dtype=int)
            hours = (candidates * sample_minutes / 60.0) % 24.0
            active_candidates = candidates[((hours >= 6.0) & (hours <= 9.5)) | ((hours >= 17.0) & (hours <= 20.5))]
            start = int(rng.choice(active_candidates)) if len(active_candidates) else int(rng.integers(0, n_steps - duration))
        else:
            start = int(rng.integers(0, n_steps - duration))
        targets[task][start : start + duration] = 1

    return targets


def _inject_missing_and_outliers(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    missing_fraction: float,
    outlier_fraction: float,
) -> pd.DataFrame:
    frame = frame.copy()
    sensor_cols = ["heart_rate", "spo2", "temperature", "light"]
    for col in sensor_cols:
        n_missing = int(len(frame) * missing_fraction)
        if n_missing:
            frame.loc[rng.choice(frame.index.to_numpy(), size=n_missing, replace=False), col] = np.nan
        n_outliers = int(len(frame) * outlier_fraction)
        if n_outliers:
            idx = rng.choice(frame.index.to_numpy(), size=n_outliers, replace=False)
            if col == "heart_rate":
                frame.loc[idx, col] = rng.choice([10.0, 350.0], size=n_outliers)
            elif col == "spo2":
                frame.loc[idx, col] = rng.choice([45.0, 110.0], size=n_outliers)
            elif col == "temperature":
                frame.loc[idx, col] = rng.choice([30.0, 45.0], size=n_outliers)
            else:
                frame.loc[idx, col] = -100.0
    return frame


def generate_synthetic_data(
    n_pets: int = 18,
    hours_per_pet: int = 48,
    sample_minutes: int = 1,
    seed: int = 42,
    missing_fraction: float = 0.004,
    outlier_fraction: float = 0.0015,
) -> pd.DataFrame:
    """Generate synthetic multi-label pet wearable data.

    Supervised labels are *pattern labels* for model development, not veterinary diagnoses.
    General anomaly is intentionally not generated as a supervised target; it is learned by
    Isolation Forest from healthy windows.
    """
    profiles = generate_pet_profiles(n_pets=n_pets, seed=seed)
    steps = int(hours_per_pet * 60 / sample_minutes)
    if steps < 30:
        raise ValueError("Generate at least 30 time steps per pet.")
    base_time = pd.Timestamp("2026-01-01 00:00:00")
    all_frames: list[pd.DataFrame] = []

    for p_idx, profile in enumerate(profiles):
        pet_rng = np.random.default_rng(seed + 1000 + p_idx)
        timestamps = pd.date_range(base_time, periods=steps, freq=f"{sample_minutes}min")
        targets = _make_multitask_schedule(steps, pet_rng, sample_minutes=sample_minutes)

        heart_rate = np.zeros(steps)
        spo2 = np.zeros(steps)
        temperature = np.zeros(steps)
        light = np.zeros(steps)
        lat = np.zeros(steps)
        lon = np.zeros(steps)
        latent_activity = np.zeros(steps)
        lat[0], lon[0] = profile.home_lat, profile.home_lon
        temp_state = profile.baseline_temp

        for i, ts in enumerate(timestamps):
            hour = ts.hour + ts.minute / 60.0
            activity = _activity_signal(hour, pet_rng)
            if targets["activity_reduction"][i]:
                activity *= 0.08 + pet_rng.uniform(0.0, 0.08)
            if targets["abnormal_resting_hr"][i]:
                activity *= 0.16
            latent_activity[i] = activity

            activity_hr_gain = 68.0 if profile.species == "dog" else 52.0
            hr = profile.baseline_hr + activity_hr_gain * activity + pet_rng.normal(0, 5.0)
            o2 = profile.baseline_spo2 - 0.8 * activity + pet_rng.normal(0, 0.45)
            target_temp = profile.baseline_temp + 0.18 * activity
            temp_state += 0.08 * (target_temp - temp_state) + pet_rng.normal(0, 0.025)
            temp = temp_state

            if targets["fever"][i]:
                temp += 1.10 + 0.22 * math.sin(i / 12.0)
                hr += 18.0
            if targets["hypoxemia"][i]:
                o2 -= 6.2 + pet_rng.uniform(0, 2.6)
                hr += 10.0
            if targets["abnormal_resting_hr"][i]:
                hr += 50.0 + pet_rng.normal(0, 6.0)
                o2 -= 0.8

            heart_rate[i] = np.clip(hr, 35.0, 280.0)
            spo2[i] = np.clip(o2, 75.0, 100.0)
            temperature[i] = np.clip(temp, 36.0, 42.0)
            light[i] = _light_from_hour(hour, pet_rng)

            if i > 0:
                speed_mps = max(0.0, 0.05 + 2.2 * activity + pet_rng.normal(0, 0.12))
                distance_m = speed_mps * sample_minutes * 60.0
                bearing = pet_rng.uniform(0, 2 * math.pi)
                lat[i] = lat[i - 1] + distance_m * math.cos(bearing) / 111_320.0
                denom = max(0.2, math.cos(math.radians(lat[i - 1])))
                lon[i] = lon[i - 1] + distance_m * math.sin(bearing) / (111_320.0 * denom)

        task_matrix = np.column_stack([targets[t] for t in SUPERVISED_TASKS])
        overall_label = (task_matrix.max(axis=1) > 0).astype(np.int8)
        risk_type = []
        for row in task_matrix:
            active = [task for task, value in zip(SUPERVISED_TASKS, row) if value]
            risk_type.append("+".join(active) if active else "healthy")

        frame = pd.DataFrame(
            {
                "pet_id": profile.pet_id,
                "timestamp": timestamps,
                "species": profile.species,
                "age_years": profile.age_years,
                "weight_kg": profile.weight_kg,
                "baseline_hr": profile.baseline_hr,
                "baseline_spo2": profile.baseline_spo2,
                "baseline_temp": profile.baseline_temp,
                "heart_rate": heart_rate,
                "spo2": spo2,
                "temperature": temperature,
                "light": light,
                "gps_lat": lat,
                "gps_lon": lon,
                "label": overall_label,
                "risk_type": risk_type,
                "latent_activity": latent_activity,
                **{f"target_{task}": targets[task] for task in SUPERVISED_TASKS},
            }
        )
        all_frames.append(
            _inject_missing_and_outliers(frame, pet_rng, missing_fraction, outlier_fraction)
        )

    return pd.concat(all_frames, ignore_index=True).sort_values(["pet_id", "timestamp"]).reset_index(drop=True)


def save_synthetic_data(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data(**kwargs)
    df.to_csv(path, index=False)
    return df
