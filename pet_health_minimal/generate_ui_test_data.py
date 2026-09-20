"""Create deterministic, synthetic CSV fixtures for manual Streamlit testing."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "test_samples"
SCENARIOS = {
    "01_normal_pattern.csv": ("DEMO_NORMAL", set()),
    "02_low_oxygen.csv": ("DEMO_OXYGEN", {"hypoxemia"}),
    "03_fever_pattern.csv": ("DEMO_FEVER", {"fever"}),
    "04_high_resting_hr.csv": ("DEMO_RESTING_HR", {"abnormal_resting_hr"}),
    "05_reduced_activity.csv": ("DEMO_ACTIVITY", {"activity_reduction"}),
    "06_combined_patterns.csv": ("DEMO_COMBINED", {"hypoxemia", "fever"}),
}


def make_recording(pet_id: str, conditions: set[str]) -> pd.DataFrame:
    # Identical noise across scenarios makes differences easier to inspect.
    rng = np.random.default_rng(20260920)
    n = 120
    time = pd.date_range("2026-09-20 07:00:00", periods=n, freq="min")
    episode = (np.arange(n) >= 30) & (np.arange(n) < 90)
    activity = np.clip(0.55 + rng.normal(0, 0.06, n), 0, 1)
    if conditions & {"abnormal_resting_hr", "activity_reduction"}:
        activity[episode] *= 0.04
    hr = 86 + 68 * activity + rng.normal(0, 3, n)
    spo2 = 98 - 0.8 * activity + rng.normal(0, 0.25, n)
    temp = 38.5 + 0.18 * activity + rng.normal(0, 0.035, n)
    if "hypoxemia" in conditions:
        spo2[episode] -= 7.0
        hr[episode] += 10.0
    if "fever" in conditions:
        temp[episode] += 1.15
        hr[episode] += 18.0
    if "abnormal_resting_hr" in conditions:
        hr[episode] += 55.0
    hour = time.hour.to_numpy() + time.minute.to_numpy() / 60.0
    light = np.maximum(0, 25 + 850 * np.sin(np.pi * (hour - 6) / 12) + rng.normal(0, 20, n))
    speed = np.maximum(0.01, 0.05 + 2.2 * activity + rng.normal(0, 0.03, n))
    bearings = rng.uniform(0, 2 * np.pi, n)
    latitude, longitude = np.zeros(n), np.zeros(n)
    latitude[0], longitude[0] = 12.9716, 77.5946
    for i in range(1, n):
        distance = speed[i] * 60
        latitude[i] = latitude[i - 1] + distance * np.cos(bearings[i]) / 111_320
        longitude[i] = longitude[i - 1] + distance * np.sin(bearings[i]) / (111_320 * np.cos(np.radians(latitude[i - 1])))
    return pd.DataFrame({
        "pet_id": pet_id, "timestamp": time,
        "species": "dog", "age_years": 5.0, "weight_kg": 24.0,
        "baseline_hr": 86.0, "baseline_spo2": 98.0, "baseline_temp": 38.5,
        "heart_rate": hr, "spo2": spo2, "temperature": temp,
        "light": light, "gps_lat": latitude, "gps_lon": longitude,
    })


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = {}
    for filename, (pet_id, conditions) in SCENARIOS.items():
        frames[filename] = make_recording(pet_id, conditions)
    frames["07_multiple_pets.csv"] = pd.concat(list(frames.values()), ignore_index=True)
    noisy = frames["03_fever_pattern.csv"].copy()
    noisy["pet_id"] = "DEMO_MISSING_VALUES"
    noisy.loc[[10, 11, 50], "heart_rate"] = np.nan
    noisy.loc[[20, 60], "spo2"] = np.nan
    noisy.loc[70, "temperature"] = 45.0
    frames["08_missing_values.csv"] = noisy
    required = ["pet_id", "timestamp", "heart_rate", "spo2", "temperature", "light", "gps_lat", "gps_lon"]
    frames["09_minimal_columns.csv"] = frames["02_low_oxygen.csv"][required].copy()
    for filename, frame in frames.items():
        frame.to_csv(OUTPUT / filename, index=False, float_format="%.8f")

    invalid_dir = OUTPUT / "invalid"
    invalid_dir.mkdir(exist_ok=True)
    normal = frames["01_normal_pattern.csv"]
    normal.iloc[:10].to_csv(invalid_dir / "too_few_readings.csv", index=False)
    normal.drop(columns="spo2").to_csv(invalid_dir / "missing_spo2_column.csv", index=False)
    normal.drop(index=50).to_csv(invalid_dir / "sampling_gap.csv", index=False)
    (OUTPUT / "README.md").write_text(
        """# Streamlit test data

All readings are fabricated, deterministic examples for testing the software.
They are not real pet records or clinical reference cases. No training files,
models, or existing reports are replaced by this script.

Open http://localhost:8501, choose **Upload CSV**, choose a file below, select
a pet, and click **Run predictions**. Expand **Condition trends** and inspect
the full timeline: the latest reading is deliberately in the recovery period.

Each single-pet file has 120 readings, one minute apart, from 07:00 to 08:59
on September 20, 2026. Default settings produce 22 windows per pet.
Abnormal patterns begin at 07:30 and end before 08:30, followed by recovery.
Window overlap and health-score smoothing spread transitions across time.

| File | What to inspect |
|---|---|
| `01_normal_pattern.csv` | Reference recording with activity and no injected condition. A model can still raise alerts. |
| `02_low_oxygen.csv` | SpO2 decreases by 7 percentage points during the episode. |
| `03_fever_pattern.csv` | Temperature rises by 1.15 C; heart rate also rises. |
| `04_high_resting_hr.csv` | Movement falls while heart rate is elevated. |
| `05_reduced_activity.csv` | Movement falls during a morning active period. |
| `06_combined_patterns.csv` | Low oxygen and elevated temperature occur together. |
| `07_multiple_pets.csv` | All six scenarios, 720 rows; switch pets and rerun predictions. |
| `08_missing_values.csv` | Five missing measurements and one invalid temperature; expect an interpolation warning, then predictions. |
| `09_minimal_columns.csv` | Only the eight required columns; expect a baseline fallback warning. Predictions can differ from the full-metadata example. |

The valid files contain no target labels. Scenario names describe injected
sensor patterns, not guaranteed model predictions or accuracy claims. Compare
condition scores and alerts over the episode, not only the latest metric.

The `invalid/` files should be rejected before prediction:

| File | Expected validation message |
|---|---|
| `too_few_readings.csv` | At least 15 readings required. |
| `missing_spo2_column.csv` | Missing required column spo2. |
| `sampling_gap.csv` | Readings must be exactly one minute apart. |

Regenerate this sample pack from the project directory:

```powershell
.\\.venv\\Scripts\\python.exe generate_ui_test_data.py
```

Regeneration overwrites only the named sample-pack files.
""", encoding="utf-8")
    print(f"Created {len(frames)} valid CSVs and 3 invalid CSVs in {OUTPUT}")


if __name__ == "__main__":
    main()
