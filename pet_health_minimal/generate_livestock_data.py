"""Generate the balanced cow/buffalo climate dataset without changing pet artifacts."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from pet_health_ai.livestock_data import save_livestock_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cows", type=int, default=15)
    parser.add_argument("--buffaloes", type=int, default=15)
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--sample-minutes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/livestock")
    args = parser.parse_args()
    frame = save_livestock_data(args.output_dir, cows=args.cows, buffaloes=args.buffaloes,
                               hours=args.hours, sample_minutes=args.sample_minutes, seed=args.seed)
    print(frame.groupby(["region", "species"]).animal_id.nunique().unstack().to_string())
    print(f"Saved {len(frame):,} readings for {frame.animal_id.nunique()} animals to {args.output_dir}")


if __name__ == "__main__":
    main()
