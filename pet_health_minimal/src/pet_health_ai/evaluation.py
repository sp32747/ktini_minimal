from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "support": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def choose_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.linspace(0.10, 0.90, 81)
    scores = [f1_score(y_true, y_prob >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def save_confusion_plot(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, path: Path, title: str) -> None:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["Healthy", "Risk"])
    ax.set_yticks([0, 1], ["Healthy", "Risk"])
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_roc_plot(y_true: np.ndarray, y_prob: np.ndarray, path: Path, title: str) -> None:
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_pr_plot(y_true: np.ndarray, y_prob: np.ndarray, path: Path, title: str) -> None:
    if len(np.unique(y_true)) < 2:
        return
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_model_evaluation(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    reports_dir: Path,
) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(y_true, y_prob, threshold=threshold)
    safe_name = model_name.lower().replace(" ", "_")
    save_confusion_plot(y_true, y_prob, threshold, reports_dir / f"{safe_name}_confusion_matrix.png", f"{model_name} Confusion Matrix")
    save_roc_plot(y_true, y_prob, reports_dir / f"{safe_name}_roc_curve.png", f"{model_name} ROC Curve")
    save_pr_plot(y_true, y_prob, reports_dir / f"{safe_name}_pr_curve.png", f"{model_name} Precision-Recall Curve")
    return metrics


def write_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def write_predictions(
    path: Path,
    pet_ids: np.ndarray,
    timestamps: np.ndarray,
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> None:
    frame = pd.DataFrame(
        {
            "pet_id": pet_ids,
            "window_end": pd.to_datetime(timestamps),
            "label": y_true,
            **{f"{name}_prob": prob for name, prob in probabilities.items()},
        }
    )
    frame.to_csv(path, index=False)
