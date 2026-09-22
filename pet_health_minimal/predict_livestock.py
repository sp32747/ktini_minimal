import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from pet_health_ai.livestock_model import load_livestock_model, predict_livestock_dataframe


def main():
    parser = argparse.ArgumentParser(description="Predict livestock conditions using the saved six-task model.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/livestock/model_bundle.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/livestock/inference_predictions.csv")
    args = parser.parse_args()
    result = predict_livestock_dataframe(pd.read_csv(args.csv), load_livestock_model(args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.tail().to_string(index=False))
    print(f"Saved {len(result):,} windows to {args.output}")


if __name__ == "__main__":
    main()
