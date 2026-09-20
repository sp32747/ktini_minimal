from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_RISK_WEIGHTS = {
    "hypoxemia": 0.28,
    "fever": 0.24,
    "abnormal_resting_hr": 0.20,
    "activity_reduction": 0.12,
    "general_anomaly": 0.16,
}


def add_longitudinal_health_score(
    predictions: pd.DataFrame,
    alpha: float = 0.25,
    risk_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add instant and longitudinal risk/health scores independently for each pet.

    The score is a product feature for screening/triage. It is not a validated clinical scale.
    """
    out = predictions.copy().sort_values(["pet_id", "window_end"]).reset_index(drop=True)
    weights = dict(DEFAULT_RISK_WEIGHTS if risk_weights is None else risk_weights)
    cols = {
        "hypoxemia": "hypoxemia_probability",
        "fever": "fever_probability",
        "abnormal_resting_hr": "abnormal_resting_hr_probability",
        "activity_reduction": "activity_reduction_probability",
        "general_anomaly": "general_anomaly_probability",
    }
    missing = [c for c in cols.values() if c not in out]
    if missing: raise ValueError(f"Missing probability columns for health score: {missing}")

    total_weight = sum(weights.values())
    weighted = sum(weights[k] * out[v].to_numpy(float) for k, v in cols.items()) / total_weight
    max_signal = np.max(np.column_stack([out[v].to_numpy(float) for v in cols.values()]), axis=1)
    out["instant_risk_index"] = np.clip(0.70 * weighted + 0.30 * max_signal, 0.0, 1.0)
    out["longitudinal_risk_index"] = (
        out.groupby("pet_id", sort=False)["instant_risk_index"]
        .transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    )
    out["health_score_0_100"] = 100.0 * (1.0 - out["longitudinal_risk_index"])
    out["screening_status"] = np.select(
        [out["health_score_0_100"] < 60, out["health_score_0_100"] < 80],
        ["ATTENTION_PATTERN", "WATCH_PATTERN"],
        default="STABLE_PATTERN",
    )
    return out
