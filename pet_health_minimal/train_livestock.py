import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from pet_health_ai.livestock_model import train_livestock_model


def main():
    parser = argparse.ArgumentParser(description="Train six livestock condition models on climate-aware sensor windows.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/livestock/synthetic_livestock_sensor_data.csv")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/livestock")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports/livestock")
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_livestock_model(args.data, args.artifacts, args.reports, args.seed, args.trees)


if __name__ == "__main__":
    main()
