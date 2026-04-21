"""
CLI entrypoint for building a ToolEQA dataset with both positive and negative samples.

Usage:
    python -m src.train.RFT.verl_adapter.prepare_dataset_with_neg \
        --source data/ToolTrajectory/trainval.json \
        --output src/train/RFT/verl_adapter/data/tooleqa_train_with_neg.jsonl \
        --neg-per-pos 2 --seed 42
"""

import argparse
import json
from pathlib import Path

from .dataset_builder import build_verl_dataset
from .dataset_builder_neg import NEG_TYPES, build_dataset_with_negatives


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ToolEQA dataset with negative samples")
    parser.add_argument("--source", required=True, help="Path to ToolEQA source JSON (trainval.json)")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Limit on positive samples")
    parser.add_argument("--neg-per-pos", type=int, default=2, help="Number of negative variants per positive sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for perturbation")
    parser.add_argument(
        "--neg-types",
        nargs="+",
        default=None,
        choices=NEG_TYPES,
        help="Subset of perturbation types to use (default: all)",
    )
    args = parser.parse_args()

    # Step 1: build positive records in verl JSONL format
    print(f"Loading positive samples from {args.source}...")
    positive_records = build_verl_dataset(args.source, "/dev/null", limit=args.limit)
    print(f"  {len(positive_records)} positive samples loaded")

    # Step 2: generate negatives
    print(f"Generating negatives (neg-per-pos={args.neg_per_pos}, seed={args.seed})...")
    all_records = build_dataset_with_negatives(
        positive_records,
        neg_types=args.neg_types,
        neg_per_pos=args.neg_per_pos,
        seed=args.seed,
    )

    # Step 3: write combined JSONL
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    neg_count = sum(1 for r in all_records if r.get("extra_info", {}).get("is_negative"))
    pos_count = len(all_records) - neg_count
    print(f"\nDataset summary:")
    print(f"  Positive: {pos_count}")
    print(f"  Negative: {neg_count}")
    print(f"  Total:    {len(all_records)}")

    # Count by perturbation type
    type_counts = {}
    for r in all_records:
        t = r.get("extra_info", {}).get("perturbation_type", "none")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  By type: {type_counts}")
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
