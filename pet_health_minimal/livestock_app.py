"""Explore livestock climates and run the separate six-condition model."""
import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from pet_health_ai.livestock_model import load_livestock_model, predict_livestock_dataframe
from pet_health_ai.livestock_features import RANGES
from pet_health_ai.livestock_data import temperature_humidity_index

DATA = ROOT / "data/livestock/synthetic_livestock_sensor_data.csv"
MODEL = ROOT / "artifacts/livestock/model_bundle.joblib"
TASK_NAMES = {"hypoxemia": "Low oxygen", "fever": "Fever pattern", "abnormal_resting_hr": "Resting heart rate",
              "activity_reduction": "Reduced activity", "heat_stress": "Heat stress", "cold_stress": "Cold stress"}


@st.cache_data(show_spinner=False)
def load_data(path: str, modified_ns: int):
    return pd.read_csv(path, parse_dates=["timestamp"])


@st.cache_resource(show_spinner=False)
def load_model(path: str, modified_ns: int):
    return load_livestock_model(path)


def main():
    st.set_page_config(page_title="KTINOSKARE | Livestock climates", page_icon="🌾", layout="wide")
    st.title("KTINOSKARE · LIVESTOCK EXPLORER")
    st.write("animal readings across five Indian climate scenarios.")
   # st.info("The Predictions tab uses six trained livestock models. sensor episode labels are generated ground truth, not model predictions. Scores are experimental and trained on synthetic data.")
    source = st.sidebar.radio("Data source", ["livestock data", "Upload CSV"], key="livestock_source")
    try:
        if source == "Upload CSV":
            uploaded = st.file_uploader("Upload livestock sensor data (labels optional)", type=["csv"], key="livestock_upload")
            if uploaded is None:
                st.info("Upload a livestock CSV including species, body sensors, ambient temperature, humidity and wind.")
                return
            payload = uploaded.getvalue()
            data_signature = hashlib.sha256(payload).hexdigest()
            frame = pd.read_csv(io.BytesIO(payload), parse_dates=["timestamp"])
        else:
            if not DATA.exists():
                st.error("Generate the dataset first: python generate_livestock_data.py")
                return
            data_signature = str(DATA.stat().st_mtime_ns)
            frame = load_data(str(DATA), DATA.stat().st_mtime_ns).copy()
        if "animal_id" not in frame and "pet_id" in frame:
            frame["animal_id"] = frame["pet_id"]
        if not {"animal_id", "species", "timestamp"}.issubset(frame) or frame.empty:
            raise ValueError("Provide animal_id, species and timestamp columns with at least one reading.")
        if frame[["animal_id", "species", "timestamp"]].isna().any().any():
            raise ValueError("Animal ID, species and timestamp cannot be empty.")
        frame["animal_id"] = frame.animal_id.astype(str)
        frame["species"] = frame.species.astype(str).str.lower().str.strip()
        if not frame.species.isin(["cow", "buffalo"]).all():
            raise ValueError("Supported species: cow and buffalo.")
        missing = set(RANGES).difference(frame.columns)
        if missing:
            raise ValueError("Missing livestock sensor columns: " + ", ".join(sorted(missing)))
        if "temperature_humidity_index" not in frame:
            frame["temperature_humidity_index"] = temperature_humidity_index(
                pd.to_numeric(frame.ambient_temperature_c, errors="coerce"),
                pd.to_numeric(frame.relative_humidity_pct, errors="coerce"))
        if "region" not in frame:
            frame["region"] = frame["region_id"] if "region_id" in frame else "Unspecified"
        frame["region"] = frame.region.fillna("Unspecified").astype(str)
    except (ValueError, OSError, UnicodeError) as exc:
        st.error(str(exc))
        return
    with st.sidebar:
        st.header("Select an animal")
        region = st.selectbox("Region", sorted(frame.region.unique()), key="livestock_region")
        species = st.selectbox("Species", sorted(frame.loc[frame.region.eq(region), "species"].unique()), key="livestock_species")
        candidates = frame.loc[frame.region.eq(region) & frame.species.eq(species)]
        animal = st.selectbox("Animal ID", sorted(candidates.animal_id.unique()), key="livestock_animal")
        st.caption("Each region uses a different illustrative seasonal profile. Dates are local India time; weather is simulated.")
        st.caption("The existing cat/dog model artifacts are not used in this explorer.")
    selected = candidates.loc[candidates.animal_id.eq(animal)].copy()
    first = selected.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Animals in dataset", frame.animal_id.nunique())
    cols[1].metric("Cows", frame.loc[frame.species.eq("cow"), "animal_id"].nunique())
    cols[2].metric("Buffaloes", frame.loc[frame.species.eq("buffalo"), "animal_id"].nunique())
    cols[3].metric("Readings for selected animal", f"{len(selected):,}")
    st.subheader(f"{animal} · {region}, {first.get('state', '')}")
    st.caption(f"{first.get('climate_zone', 'Climate observations')} | {selected.timestamp.min()} to {selected.timestamp.max()} | Asia/Kolkata")
    predictions_tab, environment, animal_tab, episodes, table = st.tabs(["Predictions", "Climate", "Animal readings", "Synthetic episodes", "Data & downloads"])
    indexed = selected.set_index("timestamp")
    with predictions_tab:
        st.caption("Uses 15 one-minute readings per window, advancing by 5 readings. Invalid sensor values are filled from prior readings; missing baseline/age/weight columns use species defaults.")
        model_signature = MODEL.stat().st_mtime_ns if MODEL.exists() else None
        signature = (data_signature, animal, model_signature)
        if st.session_state.get("livestock_prediction_signature") != signature:
            st.session_state.pop("livestock_predictions", None)
        if model_signature is None:
            st.warning("Train livestock models first: python train_livestock.py")
        else:
            try:
                bundle = load_model(str(MODEL), model_signature)
                cohort = next((name for name, ids in bundle["animal_splits"].items() if animal in ids), "not in the saved split")
                st.caption(f"Animal ID in original model split: {cohort}. Reusing an ID does not verify an uploaded animal's identity.")
                if st.button("Run livestock predictions", type="primary", key="run_livestock_predictions"):
                    st.session_state.pop("livestock_predictions", None)
                    with st.spinner("Scoring livestock windows..."):
                        st.session_state["livestock_predictions"] = predict_livestock_dataframe(selected, bundle)
                    st.session_state["livestock_prediction_signature"] = signature
                if "livestock_predictions" in st.session_state:
                    result = st.session_state["livestock_predictions"]
                    latest = result.iloc[-1]
                    a, b, c = st.columns(3)
                    a.metric("Latest health score", f"{latest.health_score_0_100:.1f} / 100")
                    b.metric("Screening status", latest.screening_status.replace("_", " ").title())
                    c.metric("Latest condition alerts", int(sum(latest[f"{t}_alert"] for t in TASK_NAMES)))
                    st.caption(f"Latest window ends {latest.window_end}. Health scoring uses all six signals with per-animal smoothing; it is not clinically calibrated.")
                    st.line_chart(result.set_index("window_end")[["health_score_0_100"]])
                    signals = pd.DataFrame([{"Condition": label, "Score": latest[f"{task}_probability"],
                                             "Threshold": bundle["thresholds"][task], "Alert": bool(latest[f"{task}_alert"])}
                                            for task, label in TASK_NAMES.items()])
                    st.dataframe(signals, hide_index=True)
                    st.line_chart(result.set_index("window_end")[[f"{t}_probability" for t in TASK_NAMES]])
                    st.dataframe(result, hide_index=True)
                    st.download_button("Download livestock predictions", result.to_csv(index=False), f"{animal}_predictions.csv", "text/csv")
                    st.caption("Risk history restarts for each batch. Inspect the whole timeline, not only the latest score.")
                else:
                    st.info("Click Run livestock predictions for the selected animal. Rerun after switching animals.")
            except Exception as exc:
                st.error(f"Livestock prediction could not complete: {exc}")
    with environment:
        st.write("Ambient temperature (°C)")
        st.line_chart(indexed[[c for c in ["ambient_temperature_c"] if c in indexed]])
        a, b = st.columns(2)
        with a:
            st.write("Relative humidity (%)")
            st.line_chart(indexed[[c for c in ["relative_humidity_pct"] if c in indexed]])
        with b:
            st.write("Temperature-humidity index (THI)")
            st.line_chart(indexed[[c for c in ["temperature_humidity_index"] if c in indexed]])
        st.caption("THI is a heat-exposure index. Cold scenarios use separate temperature, wind and shelter assumptions.")
    with animal_tab:
        channels = [c for c in ["heart_rate", "temperature", "spo2", "gps_speed_mps", "respiration_rate_bpm", "light"] if c in indexed]
        if channels:
            sensor = st.selectbox("Sensor channel", channels)
            st.line_chart(indexed[[sensor]])
        st.caption("temperature is body/core-like temperature in °C, not ambient or wearable skin temperature. Raw charts include injected outliers and missing values.")
        st.dataframe(selected[[c for c in ["baseline_hr", "baseline_temp", "baseline_spo2", "shade_access_fraction", "cooling_access_fraction", "shelter_protection_fraction"] if c in selected]].head(1), hide_index=True)
    with episodes:
        targets = [c for c in selected if c.startswith("target_")]
        if not targets:
            st.info("No target labels supplied. Labels are not needed to run predictions.")
        else:
            st.write("1 = scenario active; 0 = inactive")
            st.line_chart(indexed[targets])
            st.dataframe(pd.DataFrame({"Synthetic condition": targets,
                                   "Labelled readings": [int(selected[c].sum()) for c in targets]}), hide_index=True)
        st.caption("Heat can elevate body temperature without a fever label. These exposure and injection rules are not clinical diagnoses.")
    with table:
        st.dataframe(selected, hide_index=True)
        st.download_button("Download selected animal CSV", selected.to_csv(index=False), f"{animal}.csv", "text/csv")
        st.write("Animal counts by region and species")
        st.dataframe(frame.groupby(["region", "species"]).animal_id.nunique().unstack())
        st.caption("Full dataset and region files are in data/livestock/. Regenerate with generate_livestock_data.py.")


if __name__ == "__main__":
    main()
