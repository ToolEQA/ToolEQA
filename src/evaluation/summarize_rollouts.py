"""Summarize VERL rollout or validation JSONL diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any


FIELDS = (
    "llm_match",
    "semantic_score",
    "score",
    "acc",
    "evidence_coverage",
    "invalid_count",
    "duplicate_count",
    "no_progress_count",
    "tool_count",
    "tool_call_accuracy",
    "tool_attempt_count",
    "successful_tool_call_count",
    "final_answer_called",
    "forced_final",
    "truncated_normal_action",
    "recall_at_5",
    "recall_at_10",
    "recall_at_15",
    "raw_recall_at_5",
    "raw_recall_at_10",
    "raw_recall_at_15",
    "epath_at_5",
    "epath_at_10",
    "epath_at_15",
    "trajectory_steps",
    "trajectory_length",
)


def summarize(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    records: list[dict[str, Any]] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    summary: dict[str, Any] = {"path": str(path.resolve()), "records": len(records)}
    for field in FIELDS:
        values = [float(record[field]) for record in records if record.get(field) is not None]
        if values:
            summary[field] = fmean(values)
    steps: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("step") is not None:
            steps.setdefault(int(record["step"]), []).append(record)
    summary["per_step"] = [
        {
            "step": step,
            **{
                field: fmean(float(record[field]) for record in group if record.get(field) is not None)
                for field in FIELDS
                if any(record.get(field) is not None for record in group)
            },
        }
        for step, group in sorted(steps.items())
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.paths], indent=2))


if __name__ == "__main__":
    main()
