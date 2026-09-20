from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pet_health_ai.config import ProjectConfig
from pet_health_ai.pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Pet Health AI multisensor models.")
    parser.add_argument("--fast", action="store_true", help="Run a smaller/faster CPU smoke test.")
    parser.add_argument("--generate-only", action="store_true", help="Generate synthetic data and exit.")
    parser.add_argument("--skip-generation", action="store_true", help="Use an existing CSV instead of generating data.")
    parser.add_argument("--data-path", type=Path, default=None, help="Input/output sensor CSV path.")
    parser.add_argument("--pets", type=int, default=None, help="Override synthetic pet count.")
    parser.add_argument("--hours", type=int, default=None, help="Override hours generated per pet.")
    parser.add_argument("--epochs", type=int, default=None, help="Override LSTM/TCN epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig()

    if args.fast:
        config.data.n_pets = 8
        config.data.hours_per_pet = 12
        config.deep.epochs = 2
        config.deep.patience = 2
        config.deep.batch_size = 128
        config.deep.lstm_hidden = 32
        config.deep.tcn_channels = (16, 32)

    if args.pets is not None:
        config.data.n_pets = args.pets
    if args.hours is not None:
        config.data.hours_per_pet = args.hours
    if args.epochs is not None:
        config.deep.epochs = args.epochs

    run_training_pipeline(
        root=ROOT,
        config=config,
        data_path=args.data_path,
        skip_generation=args.skip_generation,
        generate_only=args.generate_only,
    )


if __name__ == "__main__":
    main()
