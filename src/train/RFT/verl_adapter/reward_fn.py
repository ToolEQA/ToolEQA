from __future__ import annotations

from collections.abc import Mapping
import importlib.util
from pathlib import Path
from typing import Any, Optional

try:
    from .reward_manager import ToolEQARewardManager
except ImportError:
    _REWARD_MANAGER_PATH = Path(__file__).with_name("reward_manager.py")
    _REWARD_MANAGER_SPEC = importlib.util.spec_from_file_location("tooleqa_reward_manager", _REWARD_MANAGER_PATH)
    if _REWARD_MANAGER_SPEC is None or _REWARD_MANAGER_SPEC.loader is None:
        raise RuntimeError(f"Unable to load reward manager from {_REWARD_MANAGER_PATH}")
    _REWARD_MANAGER_MODULE = importlib.util.module_from_spec(_REWARD_MANAGER_SPEC)
    _REWARD_MANAGER_SPEC.loader.exec_module(_REWARD_MANAGER_MODULE)
    ToolEQARewardManager = _REWARD_MANAGER_MODULE.ToolEQARewardManager


REWARD_MANAGER = ToolEQARewardManager()


def _to_python(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    value = _to_python(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _normalize_trace(value: Any) -> list[dict[str, Any]]:
    value = _to_python(value)
    if not isinstance(value, list):
        return []
    normalized = []
    for step in value:
        if isinstance(step, Mapping):
            normalized.append(dict(step))
    return normalized

def _normalize_gold_answer(extra_info: dict[str, Any]) -> str:
    answer = extra_info.get("answer")
    proposals = extra_info.get("proposals") or []
    if answer in ["A", "B", "C", "D"] and proposals:
        idx = ord(answer) - ord("A")
        if 0 <= idx < len(proposals):
            return str(proposals[idx]).strip().lower()
    return str(answer).strip().lower() if answer is not None else ""


def _normalize_prediction(solution_str: str, extra_info: dict[str, Any]) -> str:
    final_answer = extra_info.get("tooleqa_final_answer")
    if final_answer is not None:
        return str(_to_python(final_answer)).strip().lower()
    return str(solution_str).strip().lower()


def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> dict[str, Any]:
    extra_info = _as_dict(extra_info)
    sample = dict(extra_info)
    trace = _normalize_trace(extra_info.get("tooleqa_trace"))
    final_answer = _to_python(extra_info.get("tooleqa_final_answer"))

    is_negative = extra_info.get("is_negative", False)
    pred = _normalize_prediction(solution_str, extra_info)
    if final_answer is None and pred:
        final_answer = pred
    if final_answer is not None:
        final_answer = str(final_answer).strip()

    gold = _normalize_gold_answer(extra_info)
    base_correct = pred == gold or pred == str(ground_truth).strip().lower()

    if is_negative:
        neg_type = extra_info.get("perturbation_type", "none")
        if neg_type == "wrong_answer":
            score = 0.3 if base_correct else 0.0
            return {
                "score": score,
                "r_ans": score,
                "r_invalid": 0.0,
                "r_redund": 0.0,
                "r_prem": 0.0,
                "r_find": 0.0,
                "r_info": 0.0,
                "r_prog": 0.0,
                "r_hall": 0.0,
                "total": score,
                "details": {"negative_sample": True, "perturbation_type": neg_type},
            }
        return {
            "score": 0.0,
            "r_ans": 0.0,
            "r_invalid": 0.0,
            "r_redund": 0.0,
            "r_prem": 0.0,
            "r_find": 0.0,
            "r_info": 0.0,
            "r_prog": 0.0,
            "r_hall": 0.0,
            "total": 0.0,
            "details": {"negative_sample": True, "perturbation_type": neg_type},
        }

    breakdown = REWARD_MANAGER.compute(sample=sample, trace=trace, final_answer=final_answer)
    # Keep a safe fallback for truncated trajectories that never emitted FinalAnswer.
    if final_answer is None and base_correct:
        breakdown.r_ans = max(breakdown.r_ans, REWARD_MANAGER.answer_reward)
        breakdown.total = (
            breakdown.r_ans
            + breakdown.r_invalid
            + breakdown.r_redund
            + breakdown.r_prem
            + breakdown.r_find
            + breakdown.r_info
            + breakdown.r_prog
            + breakdown.r_hall
        )

    return {
        "score": breakdown.total,
        "r_ans": breakdown.r_ans,
        "r_invalid": breakdown.r_invalid,
        "r_redund": breakdown.r_redund,
        "r_prem": breakdown.r_prem,
        "r_find": breakdown.r_find,
        "r_info": breakdown.r_info,
        "r_prog": breakdown.r_prog,
        "r_hall": breakdown.r_hall,
        "total": breakdown.total,
        "details": breakdown.details,
        "trace_length": breakdown.details.get("trace_length", len(trace)),
        "num_turns": _to_python(extra_info.get("num_turns")),
    }
