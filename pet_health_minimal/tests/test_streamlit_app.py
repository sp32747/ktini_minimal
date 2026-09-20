from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "artifacts/multitask_lstm_model.pt").exists(), reason="Saved model artifacts required")
def test_real_models_from_pasted_csv_and_stale_result_clearing():
    # Unlabelled input exercises the same preparation path as a CSV upload.
    raw = pd.read_csv(ROOT / "data/synthetic_pet_sensor_data.csv").iloc[:20]
    columns = ["pet_id", "timestamp", "heart_rate", "spo2", "temperature", "light", "gps_lat", "gps_lon"]
    payload = raw[columns].to_csv(index=False)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not app.exception
    app.radio(key="source").set_value("Paste CSV").run()
    app.text_area(key="pasted_csv").set_value(payload).run()
    assert not app.error
    app.button(key="run_predictions").click().run()
    assert not app.exception
    assert not app.error
    predictions = app.session_state["predictions"]
    assert len(predictions) == 2
    assert predictions.health_score_0_100.between(0, 100).all()
    assert any(metric.label == "Latest health score" for metric in app.metric)
    # Changing data must not show a previous batch's score.
    raw.loc[0, "heart_rate"] += 1
    app.text_area(key="pasted_csv").set_value(raw[columns].to_csv(index=False)).run()
    assert not any(metric.label == "Latest health score" for metric in app.metric)
    app.text_area(key="pasted_csv").set_value("pet_id,timestamp\nP1,invalid\n").run()
    assert app.error
    assert not app.exception


def test_upload_empty_state():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    app.radio(key="source").set_value("Upload CSV").run()
    assert not app.exception
    assert any("Upload a CSV" in item.value for item in app.info)
