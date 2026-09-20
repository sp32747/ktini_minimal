from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class DataConfig:
    n_pets: int = 18
    hours_per_pet: int = 48
    sample_minutes: int = 1
    seed: int = 42
    missing_fraction: float = 0.004
    outlier_fraction: float = 0.0015


@dataclass
class FeatureConfig:
    window_minutes: int = 15
    stride_minutes: int = 5
    positive_window_fraction: float = 0.30


@dataclass
class DeepModelConfig:
    batch_size: int = 128
    epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    lstm_hidden: int = 64
    tcn_channels: tuple[int, ...] = (32, 64, 64)
    dropout: float = 0.20


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    deep: DeepModelConfig = field(default_factory=DeepModelConfig)
    train_fraction: float = 0.60
    val_fraction: float = 0.20
    random_state: int = 42

    def to_dict(self) -> dict:
        d = asdict(self)
        d["deep"]["tcn_channels"] = list(self.deep.tcn_channels)
        return d


def default_paths(root: Path) -> dict[str, Path]:
    return {
        "data": root / "data" / "synthetic_pet_sensor_data.csv",
        "artifacts": root / "artifacts",
        "reports": root / "reports",
    }
