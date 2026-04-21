import argparse
from pathlib import Path

from .dataset_builder import build_verl_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/ToolTrajectory/trainval.json",
        help="Source ToolEQA dataset.",
    )
    parser.add_argument(
        "--output",
        default="src/train/RFT/verl_adapter/data/tooleqa_train.jsonl",
        help="Output jsonl path for verl training.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    records = build_verl_dataset(args.source, args.output, limit=args.limit)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
