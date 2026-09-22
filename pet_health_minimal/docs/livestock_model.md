# Livestock prediction model

The livestock pipeline trains six independent XGBoost binary classifiers:
low oxygen, fever, abnormal resting heart rate, reduced activity, heat stress,
and cold stress. Multiple conditions can be active together. This is a tabular
baseline, separate from the pet pipeline's LSTM/TCN/ensemble models. There is no
livestock Isolation Forest or general-anomaly output in this version.

## Run

```powershell
.\.venv\Scripts\python.exe train_livestock.py
.\.venv\Scripts\python.exe -m streamlit run livestock_app.py --server.port 8502
.\.venv\Scripts\python.exe predict_livestock.py data/livestock/livestock_inference_input.csv
```

In the app, choose region, species and animal, open **Predictions**, and click
**Run livestock predictions**. Switching animals clears old results. Generated
ground-truth labels remain in the **Synthetic episodes** tab. **Upload CSV**
accepts unlabelled inputs; downloads contain per-window predictions.

## Inputs and features

Required columns: `animal_id` (or `pet_id`), `species` (`cow`/`buffalo`),
`timestamp`, `heart_rate`, `spo2`, `temperature`, `light`, `gps_lat`, `gps_lon`,
`ambient_temperature_c`, `relative_humidity_pct`, `wind_speed_mps`.

Use at least 15 consecutive readings per animal exactly one minute apart.
Naive timestamps represent India local time; timezone-aware timestamps are
converted to Asia/Kolkata. Missing minutes and duplicates are rejected. Units:
bpm, SpO2 percent, body/ambient Celsius, lux, decimal-degree GPS, RH percent,
and wind in m/s. Body temperature is distinct from ambient temperature.

Optional baselines, age and weight use species defaults if columns are absent.
If columns are supplied, every value must be finite. GPS and environmental
channels need valid initial readings. Sensor values outside coded limits become
missing and are forward-filled per animal, never interpolated using future
observations. Initial missing vital values use baseline defaults; all-invalid
channels are rejected. Long sensor failures need additional quality handling
before production use.

`livestock_features.py` creates 15-row windows every 5 rows. It aggregates means,
standard deviations, minima, maxima and slopes of vital/environmental variables,
GPS speed, baseline deviations and 30-minute historical climate values. It also
includes time of day, inactivity, elevated HR while inactive, species indicators,
age, weight and baselines. THI and GPS speed are recalculated from raw inputs.
Respiration, shade/cooling/shelter metadata and simulator quality flags are not
used as model inputs. The exact feature order is saved in the model manifest.

Animal IDs, region IDs, timestamps, all targets, overall label, risk type and
injected scenario names are excluded from the feature matrix. Region IDs are
used only for split construction and grouped evaluation.

## Training and evaluation

Each classifier defaults to 250 trees, depth 4, learning rate 0.05, histogram
training and class-imbalance weights calculated on training animals. Use
`--trees`, `--seed`, `--data`, `--artifacts` or `--reports` to override defaults.

One animal per region is assigned to validation and another to test. The other
four animals per region train the models: **20 train / 5 validation / 5 test**.
A deterministic search uses label/species coverage for stratification before
fitting; it never uses model performance to choose a split. Every split must
contain both species and both classes for all six tasks. A positive target
requires at least 30% of a window's row-level labels to be active.

Validation F1 selects each alert threshold from 0.10 to 0.90. The held-out test
set is evaluated once using those thresholds. Reports include precision,
recall, F1, ROC AUC, average precision and confusion matrices, plus breakdowns
by species and region. Group AUC is null where only one class is represented.

Artifacts: `artifacts/livestock/model_bundle.joblib` and `model_manifest.json`.
Reports: `reports/livestock/metrics.json`, `metrics_summary.csv`,
`feature_importance.csv`, `test_predictions.csv` and CLI `inference_predictions.csv`.
The manifest records feature version/order, windows, task order, thresholds,
score weights, source path, seed and animal splits. Load only trusted local
joblib artifacts. Training does not modify dog/cat models.

## Experimental health score

Weights: low oxygen 0.20, fever 0.18, resting HR 0.16, reduced activity 0.10,
heat stress 0.22, cold stress 0.14.

```text
instant risk = 0.70 × weighted mean + 0.30 × strongest condition score
smoothed risk[t] = 0.25 × instant risk[t] + 0.75 × previous smoothed risk
health score = 100 × (1 − smoothed risk)
```

The first observation initializes risk; smoothing is independent per animal and
resets for each batch. Scores below 60 are `ATTENTION_PATTERN`, 60 to below 80
are `WATCH_PATTERN`, and 80 or more are `STABLE_PATTERN`. These weights and
cutoffs are engineering defaults, not clinically validated scoring rules.

## Limits

All training labels come from the simulator. Heat/cold labels are generated
from climate exposure that is also available to the classifiers, so high
synthetic scores largely measure how well models reproduce those rules.
Animals are disjoint across splits, but regional weather is shared: test
results do not measure performance in unseen regions, seasons or real farms.
Only five animals are in the test set. Probabilities are not calibrated and
breed, health history, device quality and farm management are incompletely
represented. Before operational use, evaluate independent real records with
reviewed outcomes and held-out farms, climates and time periods.
