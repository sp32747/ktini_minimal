# Cows and buffaloes across five Indian climate scenarios

This dataset contains **15 cows and 15 buffaloes total**, with **3 of each species
per region**. Each adult female animal has **72 hours of one-minute readings**:
4,320 rows per animal and **129,600 rows overall**. Region dates intentionally
represent different seasons; these are not simultaneous weather observations.

| Region | Illustrative scenario | Nominal daily ambient envelope | Mean RH assumption | Start date |
|---|---|---|---|---|
| Jaisalmer, Rajasthan | Hot and dry | 24–44 °C | 25% | 2026-05-01 |
| Thrissur, Kerala | Hot and humid | 25–34 °C | 85% | 2026-07-01 |
| Ludhiana, Punjab | Warm plains with daily variation | 18–37 °C | 50% | 2026-05-01 |
| Bengaluru, Karnataka | Mild plateau | 16–29 °C | 60% | 2026-02-01 |
| Srinagar, Jammu and Kashmir | Cold winter | −6–10 °C | 70% | 2026-01-01 |

Daily offsets and smooth noise can take generated temperatures slightly beyond
the nominal envelopes. The coordinates represent hypothetical farm areas, not
actual livestock locations. These five scenarios do not capture all Indian
climates, seasons, or livestock populations.

## Files

- `synthetic_livestock_sensor_data.csv`: full labelled data.
- `livestock_inference_input.csv`: observations and animal metadata without
  targets, injected episode names, or simulator quality flags. Intended for a
  separate trained livestock model, not the saved dog/cat models.
- `animal_profiles.csv`: one row per animal, including baseline and management parameters.
- `region_species_summary.csv`: counts, climate ranges, and synthetic stress fractions.
- `generation_metadata.json`: configuration, formulas, and limitations.
- `regions/*_livestock.csv`: five labelled files, each containing six animals.
- `petcare/data_synthesizer.ipynb` (from the repository root): notebook for
  generation, checks, and charts.

## Columns and modeling assumptions

`animal_id` is the persistent ID; `pet_id` is an identical compatibility alias.
Both `cow` and `buffalo` are explicit species names. Dates are timezone-naive
**local India time** and include `timezone=Asia/Kolkata` metadata.

Sensor inputs include heart rate (bpm), SpO2 (%), body temperature (°C), light
(simulated lux), GPS coordinates (degrees), and derived GPS speed (m/s).
Ambient temperature is a **separate** `ambient_temperature_c` column; it must
not be substituted for body temperature. Extra inputs include relative humidity
(%), wind (m/s), and idealized respiration rate (breaths/min). Respiration and
environmental measurements require corresponding real sensors or external data;
they are not measurements provided by the original five-sensor prototype.

THI combines ambient temperature and humidity:

```text
THI = 0.8 × T + (RH / 100) × (T − 14.4) + 46.4
```

THI is used for heat exposure only. Cold exposure uses a separate illustrative
temperature/wind/shelter rule. Regional weather is shared by animals in the same
region, while body responses vary with individual baselines, sensitivity,
shade, cooling, and shelter. Temperature peaks in the afternoon. Movement and
heart rate follow activity; GPS stays inside a bounded simulated farm area.

Adult baseline distributions are chosen engineering assumptions: heart rate
centers at 64 bpm for cows and 61 bpm for buffaloes; body temperature centers at
38.5 °C and 38.2 °C respectively. Individual variation is added. The buffalo
heat-response multiplier (1.12), all response coefficients, management effects,
and alert cutoffs are **not clinically calibrated**. SpO2 is idealized, with no
altitude, coat, perfusion, or device-quality model. Body temperature is a
core-like proxy, not a prediction of wearable skin temperature.

## Targets

Six independent binary targets can overlap:

```text
target_hypoxemia
target_fever
target_abnormal_resting_hr
target_activity_reduction
target_heat_stress
target_cold_stress
```

The first four describe scheduled synthetic episodes. Animals are assigned
reference, single-condition, or overlapping episodes; each scheduled episode
lasts two hours during morning activity on each simulated day. This deliberately
enriched distribution is not a disease-prevalence estimate. The heat/cold labels
come from persistent simulated exposure. **Heat-induced elevation of body
temperature does not automatically set the fever target.** Activity reduction
targets represent injected episodes; climate can also reduce movement without
setting this separate target. `label` is the maximum of all six targets.

About 0.2% missing and 0.05% outlier readings are inserted independently per
heart-rate, SpO2, temperature, and light channel, rounded down per animal.
`sensor_missing_count` and `sensor_outlier_count` record how many measurements
in that row were corrupted. These flags, `risk_type`, `injected_scenario`, all
targets, and `label` must be excluded from prediction inputs.

## Generate and explore

```powershell
.\.venv\Scripts\python.exe generate_livestock_data.py
.\.venv\Scripts\python.exe -m streamlit run livestock_app.py
```

For 15 cows and 15 buffaloes **in each region** instead (150 animals total):

```powershell
.\.venv\Scripts\python.exe generate_livestock_data.py --cows 75 --buffaloes 75
```

The notebook uses the same generator. Select a Python environment with this
project's dependencies and a Jupyter Python kernel in your editor. Generation
overwrites only the named livestock outputs in the selected output directory.
The app displays observations and known synthetic labels separately from the
**Predictions** tab, which uses six livestock classifiers. Click **Run livestock
predictions** after selecting an animal; upload unlabelled readings with the
**Upload CSV** option. Train the models with `python train_livestock.py`.

## Livestock model training

Use `train_livestock.py`, which calls `livestock_features.py` and
`livestock_model.py`. It encodes cow/buffalo separately, adds environmental
features, and fits six XGBoost classifiers. Models and thresholds are saved in
`artifacts/livestock/`; metrics, feature importance and predictions are in
`reports/livestock/`. The original pet pipeline remains separate.

Each region contributes one validation animal and one test animal; remaining
animals train the models (20/5/5 with the default dataset). THI and climate
history are prediction inputs; region IDs, targets and injected-scenario fields
are not. Exposure-derived targets make synthetic heat/cold metrics particularly
easy; real evaluation needs independent outcomes and held-out regions/time
periods. See `docs/livestock_model.md` for full details.

## Scientific context (not calibration data)

The THI formula and the importance of temperature, humidity and differing animal
responses are informed by [Mandal et al., ICAR-NDRI, 2023](https://epubs.icar.org.in/index.php/IJAnS/article/download/119779/52018/379780).
That study concerns Jersey-crossbred cows and does not validate this generator
for all breeds. [Sethi et al., ICAR, 1994](https://epubs.icar.org.in/index.php/IJAnS/article/view/31207)
studied shelter and cooling effects in Murrah buffaloes. These studies motivate
the choice of variables; their results have not been fitted to the simulator.
Regional weather envelopes, cold effects, and breed-independent coefficients
are explicitly invented scenarios, not sourced local weather normals.
