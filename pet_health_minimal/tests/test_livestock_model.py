from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pet_health_ai.livestock_features import create_livestock_windows
from pet_health_ai.livestock_model import add_livestock_score, load_livestock_model, predict_livestock_dataframe
from pet_health_ai.livestock_data import LIVESTOCK_TASKS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def readings():
    return pd.read_csv(ROOT / "data/livestock/synthetic_livestock_sensor_data.csv", nrows=50)


def test_no_target_id_or_simulator_feature_leakage(readings):
    first = create_livestock_windows(readings)
    modified = readings.copy()
    for col in modified:
        if col.startswith("target_") or col in ["label", "risk_type", "injected_scenario", "sensor_missing_count", "sensor_outlier_count"]:
            modified[col] = 999
    modified["animal_id"] = "DIFFERENT_ID"
    modified["region_id"] = "OTHER_REGION"
    second = create_livestock_windows(modified)
    pd.testing.assert_frame_equal(first.features, second.features)
    assert not any("target" in col or "region" in col or "animal_id" in col for col in first.features)


def test_preprocessing_does_not_use_future_readings(readings):
    readings.loc[12:17, "heart_rate"] = np.nan
    first = create_livestock_windows(readings.iloc[:30])
    changed = readings.copy()
    changed.loc[30:, "heart_rate"] = 200
    later = create_livestock_windows(changed)
    pd.testing.assert_frame_equal(first.features, later.features.iloc[:len(first.features)].reset_index(drop=True))


def test_climate_and_species_are_explicit_features(readings):
    before = create_livestock_windows(readings).features
    readings["ambient_temperature_c"] += 5
    readings["species"] = "cow"
    after = create_livestock_windows(readings).features
    assert after.species_cow.eq(1).all() and after.species_buffalo.eq(0).all()
    assert not np.allclose(before.thi_mean, after.thi_mean)


@pytest.mark.parametrize("problem", ["gap", "duplicate", "species", "missing_climate"])
def test_invalid_prediction_inputs_rejected(readings, problem):
    if problem == "gap":
        readings = readings.drop(index=10)
    elif problem == "duplicate":
        readings.loc[1, "timestamp"] = readings.loc[0, "timestamp"]
    elif problem == "species":
        readings["species"] = "dog"
    else:
        readings = readings.drop(columns="relative_humidity_pct")
    with pytest.raises(ValueError):
        create_livestock_windows(readings)


def test_saved_model_unlabelled_prediction_and_disjoint_splits(readings):
    bundle = load_livestock_model(ROOT / "artifacts/livestock/model_bundle.joblib")
    split = {key: set(value) for key, value in bundle["animal_splits"].items()}
    assert not split["train"] & split["test"]
    assert not split["train"] & split["validation"]
    assert not split["test"] & split["validation"]
    assert sum(map(len, split.values())) == 30
    unlabelled = readings.drop(columns=[c for c in readings if c.startswith("target_") or c in ["label", "risk_type", "injected_scenario"]])
    prediction = predict_livestock_dataframe(unlabelled, bundle)
    assert len(prediction) == 8
    assert prediction.health_score_0_100.between(0, 100).all()
    for task in LIVESTOCK_TASKS:
        assert prediction[f"{task}_probability"].between(0, 1).all()
    pd.testing.assert_frame_equal(prediction, predict_livestock_dataframe(readings, bundle))


def test_risk_history_is_independent_for_each_animal():
    data = pd.DataFrame({"animal_id": ["A", "A", "B"], "window_end": pd.date_range("2026-01-01", periods=3, freq="5min")})
    for task in LIVESTOCK_TASKS:
        data[f"{task}_probability"] = [1., 1., 0.]
    scored = add_livestock_score(data)
    assert scored.loc[scored.animal_id.eq("B"), "health_score_0_100"].iloc[0] == 100
