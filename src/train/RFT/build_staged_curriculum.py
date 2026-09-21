"""Build a deterministic shortest-episode evidence curriculum."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from src.train.RFT.dataset import semantic_task_key
from src.train.RFT.evidence import evidence_target_issues, resolve_evidence_targets


LEVELS = (
    (
        "visual_grounding",
        ("attribute-color", "attribute-special", "relationship-relationship", "status-status"),
    ),
    (
        "single_object_3d",
        ("counting-counting", "location-location", "location-special"),
    ),
    ("pairwise_3d", ("attribute-size", "distance-distance")),
)


def question_type(row: dict[str, Any]) -> str:
    return str((row.get("extra_info") or {}).get("question_type", row.get("question_type", "")))


def take_round_robin(queues: dict[str, deque[dict[str, Any]]], names: Iterable[str], count: int):
    selected: list[dict[str, Any]] = []
    names = tuple(names)
    while len(selected) < count and any(queues[name] for name in names):
        for name in names:
            if queues[name]:
                selected.append(queues[name].popleft())
                if len(selected) == count:
                    break
    return selected


def build_curriculum(
    rows: list[dict[str, Any]], *, warmup_per_level: int = 10
) -> list[dict[str, Any]]:
    if warmup_per_level <= 0:
        raise ValueError("warmup_per_level must be positive")
    mismatches: list[tuple[int, list[str]]] = []
    seen_tasks: dict[str, int] = {}
    for index, row in enumerate(rows):
        info = row.get("extra_info") or {}
        issues = list(info.get("reward_audit") or [])
        issues.extend(evidence_target_issues(row))
        if info.get("reward_eligible") is not True:
            issues.append("missing-or-false-reward-eligible")
        if not info.get("evidence_targets"):
            issues.append("missing-evidence-targets")
        try:
            resolve_evidence_targets(info, require_explicit=True)
        except ValueError as error:
            issues.append(f"invalid-audit-fields:{error}")
        task_key = semantic_task_key(row)
        if task_key in seen_tasks:
            issues.append(f"semantic-duplicate:row-{seen_tasks[task_key] + 1}")
        else:
            seen_tasks[task_key] = index
        if issues:
            mismatches.append((index, list(dict.fromkeys(issues))))
    if mismatches:
        preview = ", ".join(f"row {index + 1}: {names}" for index, names in mismatches[:5])
        raise ValueError(f"Curriculum input is not reward-safe ({preview})")

    queues: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in rows:
        queues[question_type(row)].append(row)

    expected = {name for _, names in LEVELS for name in names}
    missing = sorted(expected - set(queues))
    if missing:
        raise ValueError(f"Curriculum input is missing question types: {missing}")

    ordered: list[dict[str, Any]] = []
    for level_index, (level_name, names) in enumerate(LEVELS, start=1):
        for row in take_round_robin(queues, names, warmup_per_level):
            row = dict(row)
            extra = dict(row.get("extra_info") or {})
            extra["curriculum_level"] = level_index
            extra["curriculum_name"] = level_name
            row["extra_info"] = extra
            ordered.append(row)

    all_names = tuple(name for _, names in LEVELS for name in names)
    remainder = take_round_robin(queues, all_names, len(rows) - len(ordered))
    for row in remainder:
        row = dict(row)
        extra = dict(row.get("extra_info") or {})
        extra["curriculum_level"] = 4
        extra["curriculum_name"] = "mixed_consolidation"
        row["extra_info"] = extra
        ordered.append(row)

    if len(ordered) != len(rows):
        raise RuntimeError(f"Lost curriculum rows: input={len(rows)} output={len(ordered)}")
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-per-level", type=int, default=10)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    curriculum = build_curriculum(rows, warmup_per_level=args.warmup_per_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in curriculum:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
