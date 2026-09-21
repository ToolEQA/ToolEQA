"""Select a stable, seed-addressed validation subset from VERL JSONL data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from src.train.RFT.dataset import semantic_task_key
from src.train.RFT.evidence import evidence_target_issues, resolve_evidence_targets


def _record_id(record: dict[str, Any], line_index: int) -> str:
    info = record.get("extra_info") or {}
    return str(info.get("sample_id", info.get("index", line_index)))


def reward_eligible(record: dict[str, Any]) -> bool:
    info = record.get("extra_info") or {}
    eligible = (
        info.get("reward_eligible") is True
        and bool(info.get("evidence_targets"))
        and not info.get("reward_audit")
        and not evidence_target_issues(record)
    )
    if not eligible:
        return False
    try:
        resolve_evidence_targets(info, require_explicit=True)
    except ValueError:
        return False
    return True


def select_records(source: Path, count: int, seed: int) -> list[dict[str, Any]]:
    ranked: list[tuple[str, int, dict[str, Any]]] = []
    seen_tasks: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if not reward_eligible(record):
                continue
            task_key = semantic_task_key(record)
            if task_key in seen_tasks:
                continue
            seen_tasks.add(task_key)
            record_id = _record_id(record, line_index)
            digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
            ranked.append((digest, line_index, record))
    if count <= 0 or count > len(ranked):
        raise ValueError(f"count must be in [1, {len(ranked)}], got {count}")
    return [record for _, _, record in sorted(ranked)[:count]]


def select_stratified_records(
    source: Path,
    field: str,
    per_group: int,
    seed: int,
    rank_field: str | None = None,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    if per_group <= 0:
        raise ValueError(f"per_group must be positive, got {per_group}")
    groups: dict[str, list[tuple[float, str, int, dict[str, Any]]]] = {}
    seen_tasks: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if predicate is not None and not predicate(record):
                continue
            if not reward_eligible(record):
                continue
            task_key = semantic_task_key(record)
            if task_key in seen_tasks:
                continue
            seen_tasks.add(task_key)
            info = record.get("extra_info") or {}
            group = str(info.get(field, record.get(field, "unknown")))
            record_id = _record_id(record, line_index)
            digest = hashlib.sha256(f"{seed}:{group}:{record_id}".encode("utf-8")).hexdigest()
            try:
                rank = float(info.get(rank_field, record.get(rank_field))) if rank_field else 0.0
            except (TypeError, ValueError):
                rank = float("inf")
            groups.setdefault(group, []).append((rank, digest, line_index, record))

    short = {group: len(items) for group, items in groups.items() if len(items) < per_group}
    if short:
        raise ValueError(f"groups contain fewer than {per_group} records: {short}")
    selected: list[dict[str, Any]] = []
    for group in sorted(groups):
        selected.extend(record for _, _, _, record in sorted(groups[group])[:per_group])
    return selected


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=root / "src/train/RFT/data/validation_reward_eligible.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "src/train/RFT/data/validation_reward_eligible_fixed_50.jsonl"
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stratify-field", type=str)
    parser.add_argument("--per-group", type=int, default=1)
    parser.add_argument("--rank-field", type=str)
    parser.add_argument(
        "--require-related-mentions",
        action="store_true",
        help="exclude records whose related-object labels cannot be matched to the question",
    )
    args = parser.parse_args()

    if args.stratify_field:
        records = select_stratified_records(
            args.source,
            args.stratify_field,
            args.per_group,
            args.seed,
            rank_field=args.rank_field,
            predicate=reward_eligible
            if args.require_related_mentions
            else None,
        )
    else:
        records = select_records(args.source, args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "count": len(records),
        "seed": args.seed,
        "stratify_field": args.stratify_field,
        "per_group": args.per_group if args.stratify_field else None,
        "rank_field": args.rank_field,
        "require_related_mentions": args.require_related_mentions,
        "sample_ids": [_record_id(record, index) for index, record in enumerate(records)],
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
