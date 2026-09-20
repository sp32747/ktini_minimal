from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .features import create_window_dataset
from .health_score import add_longitudinal_health_score
from .models import MultiTaskLSTMClassifier, MultiTaskTCNClassifier, isolation_anomaly_percentile, predict_multitask_torch_proba
from .pipeline import transform_sequences


def _load_json(path: Path):
    with path.open("r",encoding="utf-8") as f: return json.load(f)


def _load_multitask_torch_model(path: Path):
    c=torch.load(path,map_location="cpu",weights_only=False); kwargs=dict(c["model_kwargs"])
    if c["model_type"]=="multitask_lstm": model=MultiTaskLSTMClassifier(c["input_dim"],c["output_dim"],**kwargs)
    elif c["model_type"]=="multitask_tcn":
        if "channels" in kwargs: kwargs["channels"]=tuple(kwargs["channels"])
        model=MultiTaskTCNClassifier(c["input_dim"],c["output_dim"],**kwargs)
    else: raise ValueError(f"Unsupported model type {c['model_type']}")
    model.load_state_dict(c["state_dict"]); model.eval(); return model,list(c["task_names"])


def predict_sensor_file(csv_path: str|Path, artifacts_dir: str|Path, window_rows=15, stride_rows=5) -> pd.DataFrame:
    return predict_sensor_dataframe(pd.read_csv(csv_path), artifacts_dir, window_rows, stride_rows)


def predict_sensor_dataframe(raw: pd.DataFrame, artifacts_dir: str|Path, window_rows=15, stride_rows=5) -> pd.DataFrame:
    """Score an in-memory sensor batch using the same path as CSV inference."""
    artifacts_dir=Path(artifacts_dir)
    windows=create_window_dataset(raw,window_minutes=window_rows,stride_minutes=stride_rows,require_label=False)
    feature_columns=_load_json(artifacts_dir/"feature_columns.json")
    missing=set(feature_columns).difference(windows.features.columns)
    if missing: raise ValueError(f"Inference features do not match training: missing {sorted(missing)}")
    x=windows.features[feature_columns].to_numpy(np.float32)

    xgb_models=joblib.load(artifacts_dir/"multitask_xgboost_models.joblib"); task_names=_load_json(artifacts_dir/"task_names.json")
    xgb=np.column_stack([xgb_models[t].predict_proba(x)[:,1] for t in task_names])

    anomaly_names=_load_json(artifacts_dir/"isolation_feature_columns.json"); anomaly_idx=[feature_columns.index(c) for c in anomaly_names]
    iso_scaler=joblib.load(artifacts_dir/"isolation_scaler.joblib"); iso_model=joblib.load(artifacts_dir/"isolation_forest.joblib"); iso_ref=np.load(artifacts_dir/"isolation_reference_scores.npy")
    anomaly=isolation_anomaly_percentile(iso_scaler,iso_model,iso_ref,x[:,anomaly_idx])

    seq=transform_sequences(joblib.load(artifacts_dir/"sequence_scaler.joblib"),windows.sequences)
    lstm,lstm_tasks=_load_multitask_torch_model(artifacts_dir/"multitask_lstm_model.pt"); tcn,tcn_tasks=_load_multitask_torch_model(artifacts_dir/"multitask_tcn_model.pt")
    if task_names!=lstm_tasks or task_names!=tcn_tasks: raise ValueError("Task order mismatch between artifacts.")
    lstm_p=predict_multitask_torch_proba(lstm,seq); tcn_p=predict_multitask_torch_proba(tcn,seq)
    ensembles=joblib.load(artifacts_dir/"task_ensemble_models.joblib"); thresholds=_load_json(artifacts_dir/"task_thresholds.json")

    result=pd.DataFrame({"pet_id":windows.pet_ids,"window_end":pd.to_datetime(windows.window_end_times),"general_anomaly_probability":anomaly})
    for j,task in enumerate(task_names):
        meta=np.column_stack([xgb[:,j],lstm_p[:,j],tcn_p[:,j],anomaly]); ens=ensembles.get(task)
        prob=ens.predict_proba(meta)[:,1] if ens is not None else 0.9*np.mean(meta[:,:3],axis=1)+0.1*meta[:,3]
        result[f"{task}_probability"]=prob; result[f"{task}_alert"]=(prob>=float(thresholds[task])).astype(int)
    result["general_anomaly_alert"]=(anomaly>=float(thresholds["general_anomaly"])).astype(int)
    return add_longitudinal_health_score(result)
