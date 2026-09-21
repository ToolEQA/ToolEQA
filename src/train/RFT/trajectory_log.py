"""Atomic, per-trajectory diagnostic logging for ToolEQA rollouts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .evidence import to_jsonable


def trajectory_log_root() -> Path | None:
    value = os.environ.get("TOOLEQA_TRAJECTORY_DIR", "").strip()
    return Path(value).expanduser().resolve() if value else None


def trajectory_path(trajectory_id: str, global_step: Any = "unknown") -> Path | None:
    root = trajectory_log_root()
    if root is None:
        return None
    try:
        step = str(int(global_step))
    except (TypeError, ValueError):
        step = "unknown"
    return root / f"step_{step}" / f"{trajectory_id}.json"


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(dict(payload)), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_trajectory(record: Mapping[str, Any]) -> str | None:
    """Write a complete rollout trace without making logging rollout-critical."""
    trajectory_id = str(record.get("trajectory_id") or uuid4().hex)
    path = trajectory_path(trajectory_id, record.get("global_step"))
    if path is None:
        return None
    payload = dict(record)
    payload["trajectory_id"] = trajectory_id
    try:
        _atomic_write(path, payload)
    except Exception as error:
        print(f"WARNING: failed to write trajectory log {path}: {type(error).__name__}: {error}")
        return None
    return trajectory_id


def attach_reward(
    trajectory_id: Any,
    global_step: Any,
    reward: Mapping[str, Any],
    *,
    path_override: Any = None,
) -> bool:
    """Atomically add the exact configured reward audit after VERL scores a rollout."""
    if not trajectory_id:
        return False
    path = Path(str(path_override)).expanduser().resolve() if path_override else trajectory_path(
        str(trajectory_id), global_step
    )
    if path is not None and not path.is_file():
        root = trajectory_log_root()
        matches = list(root.glob(f"step_*/{trajectory_id}.json")) if root is not None else []
        if len(matches) == 1:
            path = matches[0]
    if path is None or not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["reward"] = to_jsonable(dict(reward))
        _atomic_write(path, payload)
    except Exception as error:
        print(f"WARNING: failed to attach reward to trajectory log {path}: {type(error).__name__}: {error}")
        return False
    return True
