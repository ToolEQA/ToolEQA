"""Streaming conversion from EQA-RT JSON arrays to VERL JSONL records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator, Mapping

from .evidence import build_evidence_target_audit, normalize_text, to_jsonable


def iter_json_array(path: str | Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Stream objects from a top-level JSON array without loading the 777 MB file."""
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as handle:
        buffer = ""
        started = False
        finished = False
        while not finished:
            chunk = handle.read(chunk_size)
            eof = chunk == ""
            buffer += chunk
            cursor = 0

            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if not started:
                    if cursor >= len(buffer):
                        break
                    if buffer[cursor] != "[":
                        raise ValueError(f"{path} must contain a top-level JSON array")
                    started = True
                    cursor += 1
                    continue
                while cursor < len(buffer) and (buffer[cursor].isspace() or buffer[cursor] == ","):
                    cursor += 1
                if cursor >= len(buffer):
                    break
                if buffer[cursor] == "]":
                    finished = True
                    cursor += 1
                    break
                try:
                    value, end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object in {path}, got {type(value).__name__}")
                yield value
                cursor = end

            buffer = buffer[cursor:]
            if eof:
                if not finished and buffer.strip():
                    raise ValueError(f"Incomplete JSON array in {path}")
                break


def _choice_text(sample: Mapping[str, Any]) -> str:
    proposals = sample.get("proposals") or []
    if not proposals:
        return ""
    lines = ["Choices:"]
    lines.extend(f"{chr(65 + index)}. {proposal}" for index, proposal in enumerate(proposals[:4]))
    return "\n".join(lines)


def build_record(sample: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Build one VERL record; privileged annotations stay out of the prompt."""
    question = str(sample.get("question", "")).strip()
    plan = str(sample.get("plan", "")).strip()
    user_parts = [question]
    if plan:
        user_parts.extend(["Planner guidance:", plan])
    choices = _choice_text(sample)
    if choices:
        user_parts.append(choices)

    keep_fields = (
        "sample_id",
        "scene",
        "question",
        "proposals",
        "answer",
        "question_type",
        "floor",
        "floor_index",
        "init_pos",
        "init_rot",
        "related_objects",
        "traj_length",
        "plan",
    )
    extra_info = {key: sample.get(key) for key in keep_fields if key in sample}
    extra_info["index"] = index
    extra_info.update(build_evidence_target_audit(sample))
    return {
        "data_source": "tooleqa_evidence",
        "prompt": [{"role": "user", "content": "\n\n".join(part for part in user_parts if part)}],
        "reward_model": {"style": "rule", "ground_truth": sample.get("answer")},
        "agent_name": "tooleqa_evidence_agent",
        "extra_info": extra_info,
    }


def semantic_task_key(record: Mapping[str, Any]) -> str:
    """Return a stable key for the embodied task, excluding row identity.

    EQA-RT contains rows with different sample IDs that otherwise describe the
    same initial state, question, answer, and reward targets. Keeping both in a
    small GRPO curriculum silently overweights that task.
    """
    info = record.get("extra_info") or record
    fields = (
        "scene",
        "floor",
        "floor_index",
        "init_pos",
        "init_rot",
        "question",
        "proposals",
        "answer",
        "question_type",
        "related_objects",
    )
    payload = {key: to_jsonable(info.get(key)) for key in fields}
    payload["question"] = normalize_text(payload.get("question"))
    payload["question_type"] = normalize_text(payload.get("question_type"))
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def split_is_validation(sample_id: Any, ratio: float, seed: int) -> bool:
    if ratio <= 0:
        return False
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < ratio


def convert_dataset(
    source: str | Path,
    train_output: str | Path,
    val_output: str | Path,
    val_ratio: float = 0.02,
    seed: int = 42,
    limit: int | None = None,
    allowed_scenes: set[str] | None = None,
    quarantine_output: str | Path | None = None,
    val_per_question_type: int | None = None,
) -> dict[str, Any]:
    if val_per_question_type is not None and val_per_question_type <= 0:
        raise ValueError("val_per_question_type must be positive when supplied")
    if val_per_question_type is None and not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1")
    train_path = Path(train_output)
    val_path = Path(val_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path = Path(quarantine_output) if quarantine_output is not None else None
    if quarantine_path is not None:
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"source": 0, "train": 0, "validation": 0, "rejected": 0, "duplicates": 0}
    seen_tasks: dict[str, int] = {}
    rejection_reasons: Counter[str] = Counter()
    rejected_by_question_type: Counter[str] = Counter()
    eligible_records: list[dict[str, Any]] = []

    quarantine_context = (
        quarantine_path.open("w", encoding="utf-8") if quarantine_path is not None else nullcontext(None)
    )
    with train_path.open("w", encoding="utf-8") as train_handle, val_path.open(
        "w", encoding="utf-8"
    ) as val_handle, quarantine_context as quarantine_handle:
        for index, sample in enumerate(iter_json_array(source)):
            if limit is not None and counts["source"] >= limit:
                break
            if allowed_scenes is not None and sample.get("scene") not in allowed_scenes:
                continue
            record = build_record(sample, index)
            counts["source"] += 1
            info = record["extra_info"]
            task_key = semantic_task_key(record)
            if task_key in seen_tasks:
                info["reward_eligible"] = False
                info["reward_audit"] = list(info.get("reward_audit") or []) + [
                    f"semantic-duplicate:{seen_tasks[task_key]}"
                ]
                counts["duplicates"] += 1
            else:
                seen_tasks[task_key] = index
            if info.get("reward_eligible") is not True:
                counts["rejected"] += 1
                rejected_by_question_type[str(info.get("question_type", "unknown"))] += 1
                for issue in info.get("reward_audit") or []:
                    rejection_reasons[str(issue).split(":", 1)[0]] += 1
                if quarantine_path is not None:
                    assert quarantine_handle is not None
                    quarantine_handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                continue
            eligible_records.append(record)

        validation_rows: set[int] = set()
        if val_per_question_type is not None:
            groups: dict[str, list[tuple[str, int]]] = {}
            for row_index, record in enumerate(eligible_records):
                info = record["extra_info"]
                group = str(info.get("question_type", "unknown"))
                record_id = info.get("sample_id", info.get("index", row_index))
                digest = hashlib.sha256(
                    f"{seed}:{group}:{record_id}".encode("utf-8")
                ).hexdigest()
                groups.setdefault(group, []).append((digest, row_index))
            short = {
                group: len(rows)
                for group, rows in groups.items()
                if len(rows) < val_per_question_type
            }
            if short:
                raise ValueError(
                    f"question types contain fewer than {val_per_question_type} eligible records: {short}"
                )
            for rows in groups.values():
                validation_rows.update(
                    row_index for _, row_index in sorted(rows)[:val_per_question_type]
                )

        train_by_question_type: Counter[str] = Counter()
        validation_by_question_type: Counter[str] = Counter()
        for row_index, record in enumerate(eligible_records):
            info = record["extra_info"]
            if val_per_question_type is None:
                record_id = info.get("sample_id", info.get("index", row_index))
                is_val = split_is_validation(record_id, val_ratio, seed)
            else:
                is_val = row_index in validation_rows
            handle = val_handle if is_val else train_handle
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts["validation" if is_val else "train"] += 1
            counter = validation_by_question_type if is_val else train_by_question_type
            counter[str(info.get("question_type", "unknown"))] += 1
    return {
        **counts,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "rejected_by_question_type": dict(sorted(rejected_by_question_type.items())),
        "train_by_question_type": dict(sorted(train_by_question_type.items())),
        "validation_by_question_type": dict(sorted(validation_by_question_type.items())),
    }
