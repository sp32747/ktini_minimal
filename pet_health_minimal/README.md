# Pet Health AI — Multi-Task Sensor Modeling Project

A complete local Python prototype for pet-wearable health screening from:

- Heart rate
- SpO2
- Temperature
- Ambient light
- GPS

Version **0.2** evolves the original healthy/unhealthy classifier into a **multi-task time-series system** that separately estimates:

1. `hypoxemia_probability`
2. `fever_probability`
3. `abnormal_resting_hr_probability`
4. `activity_reduction_probability`
5. `general_anomaly_probability`

It then combines those signals into a **pet-specific longitudinal health score**.

> This project is an engineering/ML prototype. Synthetic labels and scores are not veterinary diagnoses or clinically validated thresholds.

## Architecture

```text
Heart rate ─────────┐
SpO2 ───────────────┤
Temperature ────────┤
Light ──────────────┤
GPS ────────────────┘
          │
          ▼
   Cleaning + GPS speed
          │
          ▼
Personal-baseline deltas
+ sliding 15-minute windows
          │
      ┌───┴──────────────────────────────────┐
      │                                      │
      ▼                                      ▼
Aggregated feature matrix              Sensor sequences
      │                                      │
      ├─ XGBoost: hypoxemia                  ├─ Shared LSTM encoder
      ├─ XGBoost: fever                      │    └─ 4 task outputs
      ├─ XGBoost: abnormal resting HR        │
      └─ XGBoost: activity reduction         └─ Shared TCN encoder
      │                                           └─ 4 task outputs
      │
      └─ Isolation Forest on healthy windows
             └─ general anomaly probability
                         │
                         ▼
             Task-specific stack ensembles
                         │
                         ▼
     4 condition probabilities + anomaly probability
                         │
                         ▼
              Instant multi-signal risk
                         │
                         ▼
             Per-pet EWMA longitudinal risk
                         │
                         ▼
              Health score from 0 to 100
```

## Why multi-task instead of one binary label?

A binary `healthy/unhealthy` score hides the reason for an alert. In this version, the product can say that the dominant pattern is, for example, low SpO2, fever-like temperature elevation, high HR while inactive, reduced activity, or an unusual combination not well represented by the supervised tasks.

The LSTM and TCN use a **shared temporal representation** but have separate output units for the four supervised tasks. XGBoost models are task-specific. Isolation Forest remains unsupervised and is trained only on windows with none of the supervised conditions active.

## Synthetic multi-label data

The generator emits raw sensor columns plus:

```text
target_hypoxemia
target_fever
target_abnormal_resting_hr
target_activity_reduction
```

Targets may overlap. For example a window can contain both hypoxemia and fever. `label` is retained as an overall `any condition active` compatibility field.

Activity-reduction episodes are injected during normally active circadian periods, so the model learns **less activity than expected in context**, rather than simply learning that nighttime inactivity is abnormal.

## Feature engineering

Examples include:

```text
heart_rate_mean / std / min / max / slope
spo2_mean / min / slope
temperature_mean / max / slope
gps_speed_mps_mean / std / max / slope
hr_delta_mean / hr_delta_max
temp_delta_mean / temp_delta_max
spo2_delta_mean / spo2_delta_min
low_spo2_fraction
inactive_fraction
active_fraction
daylight_inactive_fraction
high_hr_while_inactive
hr_speed_corr
temperature_hr_corr
hour_sin_mean / hour_cos_mean
```

Personal baseline fields are used when available. For an unseen inference file that does not contain baselines, robust per-pet medians are used as a prototype fallback.

## Longitudinal health score

For each window the system computes a weighted multi-signal risk using the four task probabilities plus general anomaly. It blends the weighted mean with the strongest current signal:

```text
instant risk = 0.70 × weighted multi-task risk
             + 0.30 × maximum individual risk signal
```

Then, independently for each pet:

```text
longitudinal_risk[t]
    = EWMA(instant_risk[t], alpha=0.25)

health_score[t]
    = 100 × (1 - longitudinal_risk[t])
```

This gives persistence: one noisy window has limited impact, while repeated abnormal windows lower the score progressively. The weights and product-status cutoffs are engineering defaults and should be calibrated on veterinary-labelled real-world data.

## Pet-level data splitting

Train, validation and test are split by **pet ID**, not random windows. A pet in the test set is never present in the training set. This reduces leakage from individual physiological baselines.

## Install

Python 3.11+ is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

## Run the fast smoke test

```bash
python run_pipeline.py --fast
```

This generates synthetic data, creates windows, trains every model, evaluates held-out pets and saves all artifacts.

## Run the fuller configuration

```bash
python run_pipeline.py
```

Optional overrides:

```bash
python run_pipeline.py --pets 24 --hours 72 --epochs 15
```

Generate data only:

```bash
python run_pipeline.py --generate-only
```

Train against an existing compatible CSV:

```bash
python run_pipeline.py --skip-generation --data-path data/my_training_data.csv
```

## Streamlit sensor testing UI

The built-in sample picker includes **Six-pet demo (07_multiple_pets.csv)**
and **Training sensor data**. All pet IDs are listed above the **Pet to inspect**
selector. Validation applies to the selected pet, so sampling gaps in another
pet's recording do not prevent switching pets. Click **Run predictions** after
switching; results are labelled with the selected pet ID.

From the project directory, install dependencies and start the app:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open http://localhost:8501. Choose **Built-in sample**, **Upload CSV**, or
**Paste CSV**, select a pet, and click **Run predictions**. The app uses the
saved models in `artifacts/` and displays health-score trends, condition scores,
alert thresholds, sensor charts, and downloadable predictions. If models are
missing, generate them with `python run_pipeline.py --fast` (this replaces
existing training artifacts and reports).

Required CSV columns:

```text
pet_id,timestamp,heart_rate,spo2,temperature,light,gps_lat,gps_lon
```

Include at least 15 consecutive readings per pet, exactly one minute apart.
Use local timestamps in one consistent timezone, heart rate in bpm, SpO2 as
a percentage (98, not 0.98), temperature in Celsius, and GPS in decimal degrees.
The UI checks missing columns, duplicate timestamps, sampling gaps, insufficient
history, and completely invalid sensor channels before scoring. Partial missing
or out-of-range sensor values are reported and interpolated by the existing
pipeline. Targets are not required for inference.

Optional metadata: `species` (`dog`/`cat`), `age_years`, `weight_kg`,
`baseline_hr`, `baseline_spo2`, and `baseline_temp`. Missing baseline columns
use per-pet medians from the submitted history. Supply historical baselines when
available. The UI fixes windows at 15 rows with a 5-row stride to match the
default one-minute training setup; use the CLI for other model configurations.

This is batch testing, not live device ingestion. Risk smoothing applies within
the selected batch and does not persist across separate uploads. Synthetic-model
scores are not clinically validated; the general anomaly score is a percentile,
not a disease probability. Uploaded data is processed in memory and is not saved
to the project's data or reports folders by the app.

## Inference

```bash
python predict.py data/synthetic_pet_sensor_data.csv
```

Default output:

```text
reports/inference_predictions.csv
```

Important output fields:

```text
pet_id
window_end
hypoxemia_probability
hypoxemia_alert
fever_probability
fever_alert
abnormal_resting_hr_probability
abnormal_resting_hr_alert
activity_reduction_probability
activity_reduction_alert
general_anomaly_probability
general_anomaly_alert
instant_risk_index
longitudinal_risk_index
health_score_0_100
screening_status
```

Product status values are deliberately phrased as pattern screening states:

```text
STABLE_PATTERN
WATCH_PATTERN
ATTENTION_PATTERN
```

They are not diagnoses.

## Saved model artifacts

```text
artifacts/
├── multitask_xgboost_models.joblib
├── multitask_lstm_model.pt
├── multitask_tcn_model.pt
├── task_ensemble_models.joblib
├── isolation_forest.joblib
├── isolation_scaler.joblib
├── isolation_reference_scores.npy
├── isolation_feature_columns.json
├── sequence_scaler.joblib
├── feature_columns.json
├── sequence_features.json
├── task_names.json
└── task_thresholds.json
```

## Reports

```text
reports/
├── multitask_metrics.json
├── multitask_test_predictions.csv
├── inference_predictions.csv
├── multitask_xgboost_feature_importance.csv
├── hypoxemia_* plots
├── fever_* plots
├── abnormal_resting_hr_* plots
├── activity_reduction_* plots
└── general_anomaly_proxy_* plots
```

Because the supplied data is synthetic and intentionally structured, test metrics can be extremely high. Those scores demonstrate that the pipeline works; they do **not** estimate performance on real pets.

## Tests

```bash
pytest -q
```

The included tests cover synthetic multi-label generation, cleaning/window creation, and longitudinal health-score behavior.

## Recommended real-product evolution

The next production step is to replace synthetic target labels with veterinarian-reviewed episodes and explicitly maintain a baseline profile for each pet. A mature system should also handle sensor-quality confidence, missing-device periods, calibration differences, breed/species context, longitudinal baseline drift, alert persistence/cool-down, and evaluation at the episode/pet level rather than only the window level.
