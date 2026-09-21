"""CLI for preparing ToolEQA evidence-GRPO train/validation files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.train.RFT.dataset import convert_dataset


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(root / "data/ToolTrajectory/trainval.json"))
    parser.add_argument("--output-dir", default=str(root / "src/train/RFT/data"))
    parser.add_argument("--train-name", default="train_reward_eligible.jsonl")
    parser.add_argument("--validation-name", default="validation_reward_eligible.jsonl")
    parser.add_argument("--quarantine-name", default="reward_quarantine.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument(
        "--val-per-question-type",
        type=int,
        default=25,
        help="Exact deterministic validation rows per question type; set 0 to use --val-ratio.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--scene-root",
        help="Only keep samples whose scene directory exists directly under this root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    allowed_scenes = None
    if args.scene_root:
        scene_root = Path(args.scene_root)
        allowed_scenes = {path.name for path in scene_root.iterdir() if path.is_dir()}
    counts = convert_dataset(
        source=args.source,
        train_output=output_dir / args.train_name,
        val_output=output_dir / args.validation_name,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit=args.limit,
        allowed_scenes=allowed_scenes,
        quarantine_output=output_dir / args.quarantine_name,
        val_per_question_type=args.val_per_question_type or None,
    )
    report = {
        "source": str(Path(args.source).expanduser().resolve()),
        "scene_root": str(Path(args.scene_root).expanduser().resolve()) if args.scene_root else None,
        "train_output": str((output_dir / args.train_name).resolve()),
        "validation_output": str((output_dir / args.validation_name).resolve()),
        "quarantine_output": str((output_dir / args.quarantine_name).resolve()),
        "val_ratio": args.val_ratio,
        "val_per_question_type": args.val_per_question_type or None,
        "seed": args.seed,
        "counts": counts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reward_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
