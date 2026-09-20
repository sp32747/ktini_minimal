from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .config import ProjectConfig
from .data_generation import SUPERVISED_TASKS, save_synthetic_data
from .evaluation import choose_threshold_by_f1, save_model_evaluation, write_metrics
from .features import WindowedDataset, create_window_dataset
from .health_score import add_longitudinal_health_score
from .models import (
    MultiTaskLSTMClassifier,
    MultiTaskTCNClassifier,
    fit_isolation_forest,
    fit_task_ensemble,
    isolation_anomaly_percentile,
    make_xgboost,
    predict_multitask_torch_proba,
    train_multitask_torch_classifier,
)


def split_pet_ids(pet_ids: np.ndarray, train_fraction=0.60, val_fraction=0.20, seed=42):
    unique = np.array(sorted(set(map(str, pet_ids))))
    if len(unique) < 5: raise ValueError("Use at least 5 unique pets for pet-level train/validation/test splitting.")
    rng = np.random.default_rng(seed); rng.shuffle(unique); n=len(unique)
    n_train=max(2,int(round(n*train_fraction))); n_val=max(1,int(round(n*val_fraction)))
    if n_train+n_val>=n: n_train=max(2,n-2); n_val=1
    return set(unique[:n_train]), set(unique[n_train:n_train+n_val]), set(unique[n_train+n_val:])


def masks_for_split(dataset: WindowedDataset, train_ids, val_ids, test_ids):
    return tuple(np.isin(dataset.pet_ids, list(ids)) for ids in (train_ids, val_ids, test_ids))


def fit_sequence_scaler(x_train: np.ndarray) -> StandardScaler:
    scaler=StandardScaler(); scaler.fit(x_train.reshape(-1,x_train.shape[-1])); return scaler


def transform_sequences(scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    shape=x.shape; return scaler.transform(x.reshape(-1,shape[-1])).reshape(shape).astype(np.float32)


def _save_torch_checkpoint(path: Path, model: torch.nn.Module, model_type: str, input_dim: int, output_dim: int, model_kwargs: dict, task_names: list[str]):
    torch.save({"model_type":model_type,"input_dim":input_dim,"output_dim":output_dim,"model_kwargs":model_kwargs,"task_names":task_names,"state_dict":model.state_dict()}, path)


def _ensure_task_split(y: np.ndarray, name: str, task_names: list[str]):
    bad=[task_names[j] for j in range(y.shape[1]) if len(np.unique(y[:,j]))<2]
    if bad: raise ValueError(f"{name} split lacks both classes for tasks {bad}. Increase history/pets or change seed.")


def _anomaly_feature_names() -> list[str]:
    return [
        "heart_rate_std","heart_rate_slope","spo2_std","spo2_min","spo2_slope",
        "temperature_std","temperature_slope","gps_speed_mps_mean","gps_speed_mps_std","gps_speed_mps_max",
        "hr_delta_mean","hr_delta_max","temp_delta_mean","temp_delta_max","spo2_delta_mean","spo2_delta_min",
        "low_spo2_fraction","inactive_fraction","active_fraction","daylight_inactive_fraction",
        "high_hr_while_inactive","hr_speed_corr","temperature_hr_corr",
    ]


def run_training_pipeline(root: Path, config: ProjectConfig, data_path: Path|None=None, skip_generation=False, generate_only=False) -> dict:
    root=Path(root); data_dir=root/"data"; artifacts_dir=root/"artifacts"; reports_dir=root/"reports"
    for d in (data_dir,artifacts_dir,reports_dir): d.mkdir(parents=True,exist_ok=True)
    data_path=Path(data_path) if data_path is not None else data_dir/"synthetic_pet_sensor_data.csv"
    if skip_generation:
        if not data_path.exists(): raise FileNotFoundError(data_path)
        raw=pd.read_csv(data_path)
    else:
        raw=save_synthetic_data(data_path,n_pets=config.data.n_pets,hours_per_pet=config.data.hours_per_pet,sample_minutes=config.data.sample_minutes,seed=config.data.seed,missing_fraction=config.data.missing_fraction,outlier_fraction=config.data.outlier_fraction)
    if generate_only: return {"data_path":str(data_path),"rows":len(raw),"pets":int(raw.pet_id.nunique())}

    window_rows=max(3,int(round(config.features.window_minutes/config.data.sample_minutes)))
    stride_rows=max(1,int(round(config.features.stride_minutes/config.data.sample_minutes)))
    dataset=create_window_dataset(raw,window_minutes=window_rows,stride_minutes=stride_rows,positive_window_fraction=config.features.positive_window_fraction)
    train_ids,val_ids,test_ids=split_pet_ids(dataset.pet_ids,config.train_fraction,config.val_fraction,config.random_state)
    train_mask,val_mask,test_mask=masks_for_split(dataset,train_ids,val_ids,test_ids)
    x=dataset.features.to_numpy(np.float32); y=dataset.task_labels; y_any=dataset.labels
    x_train,x_val,x_test=x[train_mask],x[val_mask],x[test_mask]
    y_train,y_val,y_test=y[train_mask],y[val_mask],y[test_mask]
    _ensure_task_split(y_train,"Training",dataset.task_names); _ensure_task_split(y_val,"Validation",dataset.task_names); _ensure_task_split(y_test,"Test",dataset.task_names)

    split_summary={
        "train_pet_ids":sorted(train_ids),"validation_pet_ids":sorted(val_ids),"test_pet_ids":sorted(test_ids),
        "train_windows":int(train_mask.sum()),"validation_windows":int(val_mask.sum()),"test_windows":int(test_mask.sum()),
        "task_positive_rates":{
            task:{"train":float(y_train[:,j].mean()),"validation":float(y_val[:,j].mean()),"test":float(y_test[:,j].mean())}
            for j,task in enumerate(dataset.task_names)
        }
    }

    print("[1/5] Training one XGBoost detector per condition...")
    xgb_models={}; xgb_val=np.zeros_like(y_val,dtype=float); xgb_test=np.zeros_like(y_test,dtype=float)
    importance_rows=[]
    for j,task in enumerate(dataset.task_names):
        model=make_xgboost(y_train[:,j],random_state=config.random_state+j); model.fit(x_train,y_train[:,j]); xgb_models[task]=model
        xgb_val[:,j]=model.predict_proba(x_val)[:,1]; xgb_test[:,j]=model.predict_proba(x_test)[:,1]
        importance_rows.extend({"task":task,"feature":f,"importance":float(v)} for f,v in zip(dataset.features.columns,model.feature_importances_))
    joblib.dump(xgb_models,artifacts_dir/"multitask_xgboost_models.joblib")
    pd.DataFrame(importance_rows).sort_values(["task","importance"],ascending=[True,False]).to_csv(reports_dir/"multitask_xgboost_feature_importance.csv",index=False)

    print("[2/5] Training general anomaly detector on healthy windows...")
    anomaly_names=_anomaly_feature_names(); anomaly_idx=[dataset.features.columns.get_loc(c) for c in anomaly_names]
    iso_scaler,iso_model,iso_reference=fit_isolation_forest(x_train[:,anomaly_idx],y_any[train_mask],config.random_state)
    iso_val=isolation_anomaly_percentile(iso_scaler,iso_model,iso_reference,x_val[:,anomaly_idx]); iso_test=isolation_anomaly_percentile(iso_scaler,iso_model,iso_reference,x_test[:,anomaly_idx])
    joblib.dump(iso_scaler,artifacts_dir/"isolation_scaler.joblib"); joblib.dump(iso_model,artifacts_dir/"isolation_forest.joblib"); np.save(artifacts_dir/"isolation_reference_scores.npy",iso_reference)
    (artifacts_dir/"isolation_feature_columns.json").write_text(json.dumps(anomaly_names,indent=2),encoding="utf-8")

    seq_train,seq_val,seq_test=dataset.sequences[train_mask],dataset.sequences[val_mask],dataset.sequences[test_mask]
    seq_scaler=fit_sequence_scaler(seq_train); seq_train=transform_sequences(seq_scaler,seq_train); seq_val=transform_sequences(seq_scaler,seq_val); seq_test=transform_sequences(seq_scaler,seq_test)
    joblib.dump(seq_scaler,artifacts_dir/"sequence_scaler.joblib"); input_dim=seq_train.shape[-1]; output_dim=len(dataset.task_names)

    print("[3/5] Training shared-encoder multi-task LSTM...")
    lstm=MultiTaskLSTMClassifier(input_dim,output_dim,config.deep.lstm_hidden,config.deep.dropout)
    lstm_result=train_multitask_torch_classifier(lstm,seq_train,y_train,seq_val,y_val,epochs=config.deep.epochs,batch_size=config.deep.batch_size,learning_rate=config.deep.learning_rate,weight_decay=config.deep.weight_decay,patience=config.deep.patience,seed=config.random_state)
    lstm_val=predict_multitask_torch_proba(lstm_result.model,seq_val); lstm_test=predict_multitask_torch_proba(lstm_result.model,seq_test)
    _save_torch_checkpoint(artifacts_dir/"multitask_lstm_model.pt",lstm_result.model,"multitask_lstm",input_dim,output_dim,{"hidden_dim":config.deep.lstm_hidden,"dropout":config.deep.dropout},dataset.task_names)

    print("[4/5] Training shared-encoder multi-task TCN...")
    tcn=MultiTaskTCNClassifier(input_dim,output_dim,config.deep.tcn_channels,dropout=config.deep.dropout)
    tcn_result=train_multitask_torch_classifier(tcn,seq_train,y_train,seq_val,y_val,epochs=config.deep.epochs,batch_size=config.deep.batch_size,learning_rate=config.deep.learning_rate,weight_decay=config.deep.weight_decay,patience=config.deep.patience,seed=config.random_state+1)
    tcn_val=predict_multitask_torch_proba(tcn_result.model,seq_val); tcn_test=predict_multitask_torch_proba(tcn_result.model,seq_test)
    _save_torch_checkpoint(artifacts_dir/"multitask_tcn_model.pt",tcn_result.model,"multitask_tcn",input_dim,output_dim,{"channels":list(config.deep.tcn_channels),"dropout":config.deep.dropout},dataset.task_names)

    print("[5/5] Training task-specific validation-stack ensembles and longitudinal score...")
    ensembles={}; task_val=np.zeros_like(y_val,dtype=float); task_test=np.zeros_like(y_test,dtype=float); thresholds={}
    for j,task in enumerate(dataset.task_names):
        val_meta=np.column_stack([xgb_val[:,j],lstm_val[:,j],tcn_val[:,j],iso_val]); test_meta=np.column_stack([xgb_test[:,j],lstm_test[:,j],tcn_test[:,j],iso_test])
        ens=fit_task_ensemble(val_meta,y_val[:,j],config.random_state+j); ensembles[task]=ens
        if ens is None:
            task_val[:,j]=0.9*np.mean(val_meta[:,:3],axis=1)+0.1*val_meta[:,3]; task_test[:,j]=0.9*np.mean(test_meta[:,:3],axis=1)+0.1*test_meta[:,3]
        else:
            task_val[:,j]=ens.predict_proba(val_meta)[:,1]; task_test[:,j]=ens.predict_proba(test_meta)[:,1]
        thresholds[task]=choose_threshold_by_f1(y_val[:,j],task_val[:,j])
    thresholds["general_anomaly"]=choose_threshold_by_f1(y_any[val_mask],iso_val)
    joblib.dump(ensembles,artifacts_dir/"task_ensemble_models.joblib")
    (artifacts_dir/"task_thresholds.json").write_text(json.dumps(thresholds,indent=2),encoding="utf-8")
    (artifacts_dir/"feature_columns.json").write_text(json.dumps(dataset.features.columns.tolist(),indent=2),encoding="utf-8")
    (artifacts_dir/"sequence_features.json").write_text(json.dumps(dataset.sequence_feature_names,indent=2),encoding="utf-8")
    (artifacts_dir/"task_names.json").write_text(json.dumps(dataset.task_names,indent=2),encoding="utf-8")

    metrics={"configuration":config.to_dict(),"split":split_summary,"tasks":{},"general_anomaly":{},"training":{"lstm_best_epoch":lstm_result.best_epoch,"lstm_history":lstm_result.history,"tcn_best_epoch":tcn_result.best_epoch,"tcn_history":tcn_result.history}}
    for j,task in enumerate(dataset.task_names):
        metrics["tasks"][task]=save_model_evaluation(task.replace("_"," ").title(),y_test[:,j],task_test[:,j],thresholds[task],reports_dir)
    metrics["general_anomaly"]=save_model_evaluation("General Anomaly Proxy",y_any[test_mask],iso_test,thresholds["general_anomaly"],reports_dir)
    write_metrics(metrics,reports_dir/"multitask_metrics.json")
    (reports_dir/"split_summary.json").write_text(json.dumps(split_summary,indent=2),encoding="utf-8")

    pred=pd.DataFrame({"pet_id":dataset.pet_ids[test_mask],"window_end":pd.to_datetime(dataset.window_end_times[test_mask]),"general_anomaly_probability":iso_test})
    for j,task in enumerate(dataset.task_names):
        pred[f"target_{task}"]=y_test[:,j]; pred[f"{task}_probability"]=task_test[:,j]; pred[f"{task}_alert"]=(task_test[:,j]>=thresholds[task]).astype(int)
    pred["general_anomaly_alert"]=(iso_test>=thresholds["general_anomaly"]).astype(int)
    pred=add_longitudinal_health_score(pred)
    pred.to_csv(reports_dir/"multitask_test_predictions.csv",index=False)

    print("Training complete. Multi-task metrics:")
    print(json.dumps(metrics["tasks"],indent=2))
    return metrics
