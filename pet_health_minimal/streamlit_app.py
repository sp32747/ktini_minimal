from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pet_health_ai.inference import predict_sensor_dataframe
from pet_health_ai.ui_data import REQUIRED_COLUMNS, prepare_sensor_data

ARTIFACTS = ROOT / "artifacts"
TASK_LABELS = {
    "hypoxemia": "Low oxygen", "fever": "Fever pattern",
    "abnormal_resting_hr": "Resting heart rate", "activity_reduction": "Reduced activity",
    "general_anomaly": "General anomaly",
}
STATUS_LABELS = {"STABLE_PATTERN": "Stable pattern", "WATCH_PATTERN": "Watch pattern", "ATTENTION_PATTERN": "Attention pattern"}


def main() -> None:
    st.set_page_config(page_title="KTINOSKARE", page_icon="🐾", layout="wide")
    st.title("🐾 Pet Health · KTINOSKARE")
    st.write("Explore sensor history, test the saved models, and follow each pet's health patterns.")
   # st.caption("Research prototype · Models trained on synthetic data · Scores are not veterinary diagnoses.")

    with st.sidebar:
        st.header("Test data")
        source = st.radio("Data source", ["Built-in sample", "Upload CSV", "Paste CSV"], key="source")
        sample_options = {
            "Six-pet demo (07_multiple_pets.csv)": ROOT / "data" / "test_samples" / "07_multiple_pets.csv",
            "Training sensor data": ROOT / "data" / "synthetic_pet_sensor_data.csv",
        }
        if source == "Built-in sample":
            sample_choice = st.selectbox("Sample file", list(sample_options), key="sample_file")
        st.caption("15 readings per window · New window every 5 readings · One reading per minute")
        st.download_button("Download CSV template", ",".join(REQUIRED_COLUMNS) + "\n", "sensor_template.csv", "text/csv")
        with st.expander("Input guide"):
            st.write("Use at least 15 consecutive one-minute readings per pet, with timestamps in the pet's local timezone.")
            st.code(",".join(REQUIRED_COLUMNS), language=None)
            st.write("Heart rate: bpm. SpO₂: percent (98, not 0.98). Temperature: °C. GPS: decimal degrees. Light: consistent ambient-light scale.")
            st.write("Optional: species (dog/cat), age_years, weight_kg, baseline_hr, baseline_spo2, baseline_temp. Target labels are not required.")
        st.caption("This app scores uploaded history. It does not connect to a live wearable or preserve risk across separate batches.")

    try:
        if source == "Built-in sample":
            sample = sample_options[sample_choice]
            if not sample.exists():
                st.info("Sample data is missing. Upload a CSV or generate data with python run_pipeline.py --generate-only.")
                return
            payload = sample.read_bytes()
            source_name = sample.name
        elif source == "Upload CSV":
            uploaded = st.file_uploader("Upload sensor readings", type=["csv"])
            if uploaded is None:
                st.info("Upload a CSV to preview readings and run predictions.")
                return
            payload = uploaded.getvalue()
            source_name = uploaded.name
        else:
            pasted = st.text_area("Paste CSV including the header", height=200, key="pasted_csv")
            if not pasted.strip():
                st.info("Paste a CSV with at least 15 readings per pet.")
                return
            payload = pasted.encode("utf-8")
            source_name = "Pasted CSV"
        raw = pd.read_csv(io.BytesIO(payload), dtype={"pet_id": "string"})
        if "pet_id" not in raw:
            raise ValueError("Missing required column: pet_id")
        raw["pet_id"] = raw["pet_id"].astype("string").str.strip()
        pet_ids = sorted(raw["pet_id"].dropna().loc[lambda s: s.ne("")].unique().tolist())
        if not pet_ids:
            raise ValueError("The CSV has no pet IDs. Every reading needs a pet_id.")
    except (ValueError, OSError, UnicodeError) as exc:
        st.error(str(exc))
        return

    st.subheader("Choose a pet")
    st.caption(f"Loaded: {source_name} | {len(pet_ids)} pets | {len(raw):,} readings")
    st.write("Pets in this file: " + ", ".join(pet_ids))
    if st.session_state.get("pet") not in pet_ids:
        st.session_state["pet"] = pet_ids[0]
    pet = st.selectbox("Pet to inspect", pet_ids, key="pet")
    st.caption("Select a pet, then click Run predictions. Each pet has its own readings and results.")

    # Invalidate previous results before validating a new pet or input file.
    signature = (hashlib.sha256(payload).hexdigest(), pet)
    if st.session_state.get("result_signature") != signature:
        st.session_state.pop("predictions", None)
    if raw["pet_id"].isna().any() or raw["pet_id"].eq("").any():
        st.warning("Rows without a pet ID cannot be assigned to a pet and are excluded. Add their IDs to include them.")
    try:
        readings, warnings = prepare_sensor_data(raw.loc[raw["pet_id"] == pet].copy())
    except ValueError as exc:
        st.error(f"{pet}: {exc}")
        st.info("Choose another pet above to continue testing, or correct this pet's readings.")
        return
    for warning in warnings:
        st.warning(warning)
    a, b, c = st.columns(3)
    a.metric("Pets in file", len(pet_ids))
    b.metric("Readings for this pet", f"{len(readings):,}")
    c.metric("Available windows", f"{1 + (len(readings) - 15) // 5:,}")
    st.caption(f"Recording: {readings['timestamp'].min()} → {readings['timestamp'].max()}")

    if st.button("Run predictions", type="primary", key="run_predictions"):
        st.session_state.pop("predictions", None)
        try:
            with st.spinner("Analyzing sensor windows…"):
                predictions = predict_sensor_dataframe(readings, ARTIFACTS)
                thresholds = json.loads((ARTIFACTS / "task_thresholds.json").read_text(encoding="utf-8"))
            st.session_state["predictions"] = predictions
            st.session_state["thresholds"] = thresholds
            st.session_state["result_signature"] = signature
        except Exception as exc:
            st.error(f"Prediction could not complete: {exc}")
            st.info("Check that artifacts/ contains the complete trained models. To create them, run: python run_pipeline.py --fast")

    results_tab, sensors_tab = st.tabs(["Predictions", "Sensor readings"])
    with sensors_tab:
        sensor = st.selectbox("Sensor", ["heart_rate", "spo2", "temperature", "light", "gps_lat", "gps_lon"])
        st.line_chart(readings.set_index("timestamp")[[sensor]])
        st.caption("Preview shows validated input; missing sensor values are filled during prediction.")
        st.dataframe(readings, hide_index=True, use_container_width=True)
        st.download_button("Download this pet's input", readings.to_csv(index=False), "pet_sensor_input.csv", "text/csv")

    with results_tab:
        if "predictions" not in st.session_state:
            st.info("Choose a pet and select Run predictions to see its health-score trend and alerts.")
            return
        result = st.session_state["predictions"]
        st.subheader(f"Results for {pet}")
        latest = result.iloc[-1]
        thresholds = st.session_state["thresholds"]
        a, b, c = st.columns(3)
        a.metric("Latest health score", f"{latest['health_score_0_100']:.1f} / 100")
        b.metric("Screening status", STATUS_LABELS[latest["screening_status"]])
        c.metric("Latest alerts", int(sum(latest[f"{task}_alert"] for task in TASK_LABELS)))
        st.caption(f"Latest window ends {latest['window_end']}. Stable: ≥80 · Watch: 60–<80 · Attention: <60.")
        st.subheader("Health over time")
        st.line_chart(result.set_index("window_end")[["health_score_0_100"]], color="#159A8C")
        st.subheader("Latest signals")
        signals = pd.DataFrame([
            {"Signal": label, "Score": float(latest[f"{task}_probability"]),
             "Alert threshold": thresholds[task], "Alert": bool(latest[f"{task}_alert"])}
            for task, label in TASK_LABELS.items()
        ])
        st.dataframe(signals, hide_index=True, use_container_width=True,
                     column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.3f")})
        st.caption("Condition scores are model estimates. General anomaly is a percentile relative to healthy training windows, not a disease probability. Individual alerts and overall status use different rules.")
        with st.expander("Condition trends", expanded=True):
            trend = result.set_index("window_end")[[f"{task}_probability" for task in TASK_LABELS]]
            trend.columns = list(TASK_LABELS.values())
            st.line_chart(trend)
        alerts_only = st.checkbox("Show only windows with alerts")
        display = result
        if alerts_only:
            display = result.loc[result[[f"{task}_alert" for task in TASK_LABELS]].any(axis=1)]
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.download_button("Download all predictions for this pet", result.to_csv(index=False), "pet_predictions.csv", "text/csv")


if __name__ == "__main__":
    main()
