from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_livestock_explorer_switches_regions_species_and_animals():
    app = AppTest.from_file(str(ROOT / "livestock_app.py"), default_timeout=60).run()
    assert not app.exception
    assert not app.error
    assert len(app.selectbox(key="livestock_region").options) == 5
    app.selectbox(key="livestock_region").set_value("Jaisalmer").run()
    app.selectbox(key="livestock_species").set_value("buffalo").run()
    assert len(app.selectbox(key="livestock_animal").options) == 3
    assert all(value.startswith("BUFFALO_RJ_") for value in app.selectbox(key="livestock_animal").options)
    app.selectbox(key="livestock_animal").set_value("BUFFALO_RJ_003").run()
    assert not app.exception
    assert any("BUFFALO_RJ_003" in item.value for item in app.subheader)
    assert any("not model predictions" in item.value for item in app.info)


def test_run_livestock_predictions_and_switch_animal():
    app = AppTest.from_file(str(ROOT / "livestock_app.py"), default_timeout=60).run()
    animal = app.selectbox(key="livestock_animal").value
    app.button(key="run_livestock_predictions").click().run()
    assert not app.exception
    assert not app.error
    result = app.session_state["livestock_predictions"]
    assert result.animal_id.unique().tolist() == [animal]
    assert len(result) == 862
    assert any(metric.label == "Latest health score" for metric in app.metric)
    options = app.selectbox(key="livestock_animal").options
    app.selectbox(key="livestock_animal").set_value(next(a for a in options if a != animal)).run()
    assert not any(metric.label == "Latest health score" for metric in app.metric)


def test_upload_without_synthetic_labels():
    import pandas as pd
    raw = pd.read_csv(ROOT / "data/livestock/livestock_inference_input.csv", nrows=50)
    app = AppTest.from_file(str(ROOT / "livestock_app.py"), default_timeout=60).run()
    app.radio(key="livestock_source").set_value("Upload CSV").run()
    app.file_uploader(key="livestock_upload").upload("livestock.csv", raw.to_csv(index=False).encode(), "text/csv").run()
    app.button(key="run_livestock_predictions").click().run()
    assert not app.exception
    assert not app.error
    assert len(app.session_state["livestock_predictions"]) == 8
    assert any("No target labels" in item.value for item in app.info)
