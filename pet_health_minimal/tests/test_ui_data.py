import numpy as np
import pandas as pd
import pytest

from pet_health_ai.ui_data import prepare_sensor_data


@pytest.fixture
def readings():
    return pd.DataFrame({
        "pet_id": ["001"] * 20,
        "timestamp": pd.date_range("2026-01-01 09:00", periods=20, freq="min"),
        "heart_rate": 90.0, "spo2": 98.0, "temperature": 38.5,
        "light": 300.0, "gps_lat": 12.9, "gps_lon": 77.6,
    })


def test_input_preserves_id_and_reports_missing_baselines(readings):
    readings.loc[0, "heart_rate"] = np.nan
    frame, warnings = prepare_sensor_data(readings.iloc[::-1])
    assert frame.pet_id.unique().tolist() == ["001"]
    assert frame.timestamp.is_monotonic_increasing
    assert any("interpolated" in message for message in warnings)
    assert any("baselines" in message for message in warnings)


@pytest.mark.parametrize("problem, message", [
    ("missing_column", "Missing required"),
    ("duplicate", "Duplicate"),
    ("gap", "one minute apart"),
    ("empty_sensor", "no valid readings"),
    ("short_pet", "every pet"),
    ("baseline", "Optional column"),
    ("timestamp", "valid timestamp"),
])
def test_rejects_inputs_that_could_produce_misleading_results(readings, problem, message):
    if problem == "missing_column":
        readings = readings.drop(columns="spo2")
    elif problem == "duplicate":
        readings.loc[1, "timestamp"] = readings.loc[0, "timestamp"]
    elif problem == "gap":
        readings.loc[19, "timestamp"] += pd.Timedelta(minutes=1)
    elif problem == "empty_sensor":
        readings["spo2"] = 0.98
    elif problem == "short_pet":
        readings.loc[19, "pet_id"] = "002"
    elif problem == "baseline":
        readings["baseline_hr"] = np.nan
    elif problem == "timestamp":
        readings.loc[0, "timestamp"] = pd.NaT
    with pytest.raises(ValueError, match=message):
        prepare_sensor_data(readings)
