"""EQA-RT Recall@D and EPath@D metrics used by the paper evaluator."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


DISTANCE_THRESHOLDS = (5, 10, 15)
FOV_DEGREES = 120.0


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        result = tuple(float(item) for item in value[:3])
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _navigation_poses(trace: Iterable[Mapping[str, Any]]) -> list[tuple[tuple[float, float, float], float]]:
    """Match the paper code: one pose after each successful navigation call."""
    poses: list[tuple[tuple[float, float, float], float]] = []
    for step in trace or []:
        if step.get("action_type") != "Navigate" or step.get("ok") is False:
            continue
        state = step.get("camera_state_after")
        if not isinstance(state, Mapping):
            continue
        position = _vector3(state.get("position"))
        try:
            yaw = float(state.get("yaw"))
        except (TypeError, ValueError):
            continue
        if position is not None and math.isfinite(yaw):
            poses.append((position, yaw))
    return poses


def _target_positions(sample: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    targets = sample.get("evidence_targets") or sample.get("related_objects") or []
    positions: list[tuple[float, float, float]] = []
    for target in targets:
        if isinstance(target, Mapping) and (position := _vector3(target.get("pos"))) is not None:
            positions.append(position)
    return positions


def weighted_recall(
    poses: Iterable[tuple[tuple[float, float, float], float]],
    targets: Iterable[tuple[float, float, float]],
    max_distance: float,
    fov_degrees: float = FOV_DEGREES,
) -> float:
    """Implement Eq. (5): distance-weighted visibility recall."""
    pose_list = list(poses)
    target_list = list(targets)
    if not target_list:
        return 0.0
    cosine_threshold = math.cos(math.radians(fov_degrees) / 2.0)
    best_weights: list[float] = []
    for target in target_list:
        best = 0.0
        for camera, yaw in pose_list:
            offset = tuple(target[index] - camera[index] for index in range(3))
            distance = math.sqrt(sum(value * value for value in offset))
            if distance <= 0.0:
                best = 1.0
                continue
            if distance > max_distance:
                continue
            forward = (math.sin(yaw), 0.0, -math.cos(yaw))
            cosine = sum(forward[index] * offset[index] for index in range(3)) / distance
            if cosine >= cosine_threshold:
                best = max(best, 1.0 - distance / max_distance)
        best_weights.append(best)
    return sum(best_weights) / len(best_weights)


def compute_paper_metrics(
    sample: Mapping[str, Any],
    trace: Iterable[Mapping[str, Any]],
    *,
    correct: bool,
    answer_quality: float | None = None,
) -> dict[str, float]:
    """Return both table-code metrics and the unnormalized Eq. (5) recall."""
    trace_list = list(trace or [])
    poses = _navigation_poses(trace_list)
    targets = _target_positions(sample)
    step_normalizer = math.sqrt(len(poses)) if poses else 1.0

    path_lengths: list[float] = []
    for step in trace_list:
        try:
            value = float(step.get("path_length_after"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            path_lengths.append(value)
    path_length = max(path_lengths, default=0.0)
    try:
        shortest_length = float(sample.get("traj_length", sample.get("expert_path_length", 0.0)))
    except (TypeError, ValueError):
        shortest_length = 0.0

    metrics: dict[str, float] = {
        "trajectory_steps": float(len(poses)),
        "trajectory_length": float(path_length),
    }
    for distance in DISTANCE_THRESHOLDS:
        raw_recall = weighted_recall(poses, targets, float(distance))
        # This extra normalization is absent from Eq. (5), but is present in
        # eval_results_on_json.py and therefore matches the existing tables.
        table_recall = raw_recall / step_normalizer
        if shortest_length > 0.0:
            efficiency = math.exp(shortest_length / max(path_length, shortest_length))
        else:
            efficiency = 1.0
        metrics[f"raw_recall_at_{distance}"] = float(raw_recall)
        recall = raw_recall if answer_quality is not None else table_recall
        quality = float(correct) if answer_quality is None else answer_quality
        metrics[f"recall_at_{distance}"] = float(recall)
        metrics[f"epath_at_{distance}"] = float(quality * recall * efficiency)
        if answer_quality is not None:
            metrics[f"legacy_recall_at_{distance}"] = float(table_recall)
            metrics[f"legacy_epath_at_{distance}"] = float(quality * table_recall * efficiency)
    return metrics
