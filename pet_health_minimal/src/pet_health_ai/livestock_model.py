"""Six independent livestock XGBoost models, animal-level validation and inference."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .evaluation import choose_threshold_by_f1, classification_metrics
from .livestock_data import LIVESTOCK_TASKS
from .livestock_features import FEATURE_VERSION, create_livestock_windows

SCORE_WEIGHTS = dict(zip(LIVESTOCK_TASKS, [.20, .18, .16, .10, .22, .14]))


def split_livestock_animals(dataset, seed=42):
    """One validation and one test animal per region, remaining animals train.

    Try deterministic allocations for class/species coverage before any fitting.
    Labels are used for stratification; no model score is used to select splits.
    """
    animals = dataset.metadata.drop_duplicates("animal_id")
    regions = [g.animal_id.to_numpy() for _, g in animals.groupby("region_id", sort=True)]
    if len(regions) < 2 or any(len(group) < 3 for group in regions):
        raise ValueError("Need at least two regions and three animals per region for independent splits.")
    rng = np.random.default_rng(seed)
    for attempt in range(1000):
        split = {"train": [], "validation": [], "test": []}
        for group in regions:
            ids = rng.permutation(group)
            split["validation"].append(str(ids[0]))
            split["test"].append(str(ids[1]))
            split["train"].extend(map(str, ids[2:]))
        masks = {name: dataset.metadata.animal_id.isin(ids).to_numpy() for name, ids in split.items()}
        if all(dataset.metadata.loc[mask, "species"].nunique() == 2
               and np.all(dataset.targets[mask].min(axis=0) == 0)
               and np.all(dataset.targets[mask].max(axis=0) == 1) for mask in masks.values()):
            return split, masks, attempt + 1
    raise ValueError("Could not allocate disjoint animals with both classes for every task. Add animals/label diversity.")


def add_livestock_score(predictions, weights=None, alpha=.25):
    out = predictions.sort_values(["animal_id", "window_end"]).reset_index(drop=True).copy()
    weights = SCORE_WEIGHTS if weights is None else weights
    matrix = out[[f"{task}_probability" for task in LIVESTOCK_TASKS]].to_numpy()
    weighted = sum(weights[task] * matrix[:, j] for j, task in enumerate(LIVESTOCK_TASKS)) / sum(weights.values())
    out["instant_risk_index"] = np.clip(.7 * weighted + .3 * matrix.max(axis=1), 0, 1)
    out["longitudinal_risk_index"] = out.groupby("animal_id").instant_risk_index.transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    out["health_score_0_100"] = 100 * (1 - out.longitudinal_risk_index)
    out["screening_status"] = np.select([out.health_score_0_100 < 60, out.health_score_0_100 < 80],
                                       ["ATTENTION_PATTERN", "WATCH_PATTERN"], default="STABLE_PATTERN")
    return out


def predict_window_features(dataset, bundle):
    x = dataset.features[bundle["feature_columns"]]
    result = dataset.metadata.copy()
    for task in bundle["tasks"]:
        prob = bundle["models"][task].predict_proba(x)[:, 1]
        result[f"{task}_probability"] = prob
        result[f"{task}_alert"] = (prob >= bundle["thresholds"][task]).astype(int)
    return add_livestock_score(result, bundle["score_weights"], bundle["score_alpha"])


def load_livestock_model(path):
    bundle = joblib.load(path)
    if bundle.get("feature_version") != FEATURE_VERSION or bundle.get("tasks") != list(LIVESTOCK_TASKS):
        raise ValueError("Incompatible livestock model bundle. Retrain with the current training script.")
    return bundle


def predict_livestock_dataframe(raw, bundle):
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Incompatible livestock feature version.")
    dataset = create_livestock_windows(raw, **bundle["window_config"])
    return predict_window_features(dataset, bundle)


def train_livestock_model(data_path, artifacts_dir, reports_dir, seed=42, trees=250):
    artifacts, reports = Path(artifacts_dir), Path(reports_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    if trees < 1:
        raise ValueError("trees must be positive.")
    raw = pd.read_csv(data_path)
    window_config = {"window": 15, "stride": 5, "positive_fraction": .30}
    print("Preparing livestock features...", flush=True)
    dataset = create_livestock_windows(raw, require_targets=True, **window_config)
    split, masks, attempts = split_livestock_animals(dataset, seed)
    print("Animals per split: " + str({k: len(v) for k, v in split.items()}), flush=True)
    bundle = {"feature_version": FEATURE_VERSION, "tasks": list(LIVESTOCK_TASKS),
              "feature_columns": dataset.features.columns.tolist(), "window_config": window_config,
              "models": {}, "thresholds": {}, "score_weights": SCORE_WEIGHTS, "score_alpha": .25,
              "animal_splits": split, "training_source": str(Path(data_path).resolve()), "seed": seed}
    metrics, importance = {}, []
    for j, task in enumerate(LIVESTOCK_TASKS):
        print(f"Training {task} ({j + 1}/6)...", flush=True)
        y_train = dataset.targets[masks["train"], j]
        model = XGBClassifier(n_estimators=trees, max_depth=4, learning_rate=.05, subsample=.85,
                              colsample_bytree=.9, min_child_weight=3, reg_lambda=2,
                              objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                              scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
                              random_state=seed + j, n_jobs=4)
        model.fit(dataset.features.loc[masks["train"]], y_train)
        val_prob = model.predict_proba(dataset.features.loc[masks["validation"]])[:, 1]
        threshold = choose_threshold_by_f1(dataset.targets[masks["validation"], j], val_prob)
        test_prob = model.predict_proba(dataset.features.loc[masks["test"]])[:, 1]
        bundle["models"][task], bundle["thresholds"][task] = model, threshold
        metrics[task] = classification_metrics(dataset.targets[masks["test"], j], test_prob, threshold)
        importance.extend({"task": task, "feature": name, "importance": float(value)}
                          for name, value in zip(dataset.features.columns, model.feature_importances_))

    # Persist only after all six tasks finish successfully.
    temporary = artifacts / "model_bundle.joblib.tmp"
    joblib.dump(bundle, temporary)
    temporary.replace(artifacts / "model_bundle.joblib")
    manifest = {key: value for key, value in bundle.items() if key != "models"}
    manifest["model_type"] = "Six task-specific XGBoost classifiers"
    manifest["n_estimators"] = trees
    (artifacts / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    test_dataset = type(dataset)(dataset.features.loc[masks["test"]].reset_index(drop=True),
                                dataset.metadata.loc[masks["test"]].reset_index(drop=True), dataset.targets[masks["test"]])
    predictions = predict_window_features(test_dataset, bundle)
    # Windows and prediction output share sorted animal/time order.
    for j, task in enumerate(LIVESTOCK_TASKS):
        predictions[f"target_{task}"] = test_dataset.targets[:, j]
    predictions.to_csv(reports / "test_predictions.csv", index=False)
    by_group = {}
    for group_col in ["species", "region_id"]:
        by_group[group_col] = {}
        for group_name, group in predictions.groupby(group_col):
            by_group[group_col][str(group_name)] = {
                task: classification_metrics(group[f"target_{task}"], group[f"{task}_probability"], bundle["thresholds"][task])
                for task in LIVESTOCK_TASKS}
    report = {"source_rows": len(raw), "feature_count": len(dataset.features.columns),
              "split_animals": split, "split_windows": {name: int(mask.sum()) for name, mask in masks.items()},
              "stratification_attempts": attempts, "tasks": metrics, "by_group": by_group,
              "limitations": ["Synthetic data and labels; metrics are not estimates of real-world accuracy.",
                              "Test animals are unseen, but all regions and shared regional weather appear in training.",
                              "Heat/cold targets derive from simulator exposure rules; this is not independent clinical validation.",
                              "Model probabilities and health score are not clinically calibrated."]}
    (reports / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = pd.DataFrame([{"task": task, **{key: values[key] for key in ["threshold", "precision", "recall", "f1", "roc_auc"]}} for task, values in metrics.items()])
    summary.to_csv(reports / "metrics_summary.csv", index=False)
    pd.DataFrame(importance).sort_values(["task", "importance"], ascending=[True, False]).to_csv(reports / "feature_importance.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    return report
