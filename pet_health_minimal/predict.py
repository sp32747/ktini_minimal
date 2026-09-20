from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pet_health_ai.inference import predict_sensor_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pet-health risk inference on a sensor CSV.")
    parser.add_argument("csv", type=Path, help="Sensor CSV to score.")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts", help="Trained artifact directory.")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "inference_predictions.csv")
    parser.add_argument("--window", type=int, default=15, help="Window length in rows (default: 15).")
    parser.add_argument("--stride", type=int, default=5, help="Window stride in rows (default: 5).")
    args = parser.parse_args()

    result = predict_sensor_file(
        args.csv,
        artifacts_dir=args.artifacts,
        window_rows=args.window,
        stride_rows=args.stride,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.tail(10).to_string(index=False))
    print(f"\nSaved {len(result)} window predictions to {args.output}")


if __name__ == "__main__":
    main()
