# Streamlit test data

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
.\.venv\Scripts\python.exe generate_ui_test_data.py
```

Regeneration overwrites only the named sample-pack files.
