from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


@dataclass
class TorchTrainingResult:
    model: nn.Module
    history: list[dict[str, float]]
    best_epoch: int


def make_xgboost(y_train: np.ndarray, random_state: int = 42) -> XGBClassifier:
    positives = max(1, int(np.sum(y_train == 1))); negatives = max(1, int(np.sum(y_train == 0)))
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.045, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=2.0, reg_lambda=1.2, reg_alpha=0.05,
        objective="binary:logistic", eval_metric="logloss", scale_pos_weight=negatives / positives,
        random_state=random_state, n_jobs=-1, tree_method="hist",
    )


def fit_isolation_forest(x_train, y_train, random_state=42):
    scaler = StandardScaler(); x_scaled = scaler.fit_transform(x_train); healthy = x_scaled[y_train == 0]
    if len(healthy) < 10: raise ValueError("Isolation Forest needs at least 10 healthy windows.")
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=random_state, n_jobs=-1)
    model.fit(healthy); reference_scores = np.sort(-model.score_samples(healthy))
    return scaler, model, reference_scores


def isolation_anomaly_percentile(scaler, model, reference_scores, x):
    scores = -model.score_samples(scaler.transform(x)); ranks = np.searchsorted(reference_scores, scores, side="right")
    return np.clip(ranks / max(1, len(reference_scores)), 0.0, 1.0)


class MultiTaskLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 48), nn.ReLU(), nn.Dropout(dropout), nn.Linear(48, output_dim))
    def forward(self, x):
        output, _ = self.lstm(x); return self.head(output[:, -1, :])


class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__(); padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation), Chomp1d(padding),
            nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation), Chomp1d(padding),
            nn.BatchNorm1d(out_channels), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.activation = nn.ReLU()
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x); return self.activation(self.net(x) + residual)


class MultiTaskTCNClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, channels=(32, 64, 64), kernel_size=3, dropout=0.2):
        super().__init__(); blocks = []; in_ch = input_dim
        for i, out_ch in enumerate(channels):
            blocks.append(TemporalBlock(in_ch, out_ch, kernel_size, 2**i, dropout)); in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels[-1], output_dim))
    def forward(self, x): return self.head(self.tcn(x.transpose(1, 2)))


def _predict_logits(model, x, batch_size, device):
    model.eval(); loader = DataLoader(TensorDataset(torch.as_tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False); parts=[]
    with torch.no_grad():
        for (xb,) in loader: parts.append(model(xb.to(device)).detach().cpu().numpy())
    return np.concatenate(parts)


def predict_multitask_torch_proba(model, x, batch_size=256, device=None):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); model = model.to(dev)
    logits = _predict_logits(model, x, batch_size, dev)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def _macro_auc(y_true, y_prob):
    aucs=[]
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) > 1: aucs.append(roc_auc_score(y_true[:, j], y_prob[:, j]))
    return float(np.mean(aucs)) if aucs else 0.5


def train_multitask_torch_classifier(
    model, x_train, y_train, x_val, y_val, epochs=12, batch_size=128, learning_rate=1e-3,
    weight_decay=1e-4, patience=3, seed=42, device=None,
):
    torch.manual_seed(seed); np.random.seed(seed); dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); model=model.to(dev)
    positives = np.maximum(1, y_train.sum(axis=0)); negatives = np.maximum(1, len(y_train) - positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(negatives / positives, dtype=torch.float32, device=dev))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    best_state=copy.deepcopy(model.state_dict()); best_score=-np.inf; best_epoch=0; no_improve=0; history=[]
    for epoch in range(1, epochs+1):
        model.train(); total=0.0; count=0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev); optimizer.zero_grad(set_to_none=True); logits=model(xb); loss=criterion(logits,yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); total += float(loss.item()) * len(xb); count += len(xb)
        val_prob = predict_multitask_torch_proba(model, x_val, batch_size=batch_size, device=str(dev)); score=_macro_auc(y_val,val_prob)
        history.append({"epoch": float(epoch), "train_loss": total/max(1,count), "val_macro_auc": score})
        if score > best_score + 1e-4:
            best_score=score; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); no_improve=0
        else:
            no_improve += 1
            if no_improve >= patience: break
    model.load_state_dict(best_state); return TorchTrainingResult(model=model.to("cpu"), history=history, best_epoch=best_epoch)


def fit_task_ensemble(meta_features: np.ndarray, y_val: np.ndarray, random_state: int = 42):
    if len(np.unique(y_val)) < 2: return None
    model = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=random_state, max_iter=500)
    model.fit(meta_features, y_val); return model
