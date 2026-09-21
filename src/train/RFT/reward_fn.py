"""VERL-compatible entry point for the evidence-grounded ToolEQA reward."""

from __future__ import annotations

from typing import Any, Mapping

from src.train.RFT.reward import compute_reward
from src.train.RFT.trajectory_log import attach_reward
from src.train.RFT.evidence import extract_final_answer


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _extract_runtime_fields(
    extra_info: Any,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], Any, Mapping[str, Any]]:
    info = _mapping(extra_info)
    tool_fields = _mapping(info.get("tool_extra_fields"))
    if not tool_fields:
        tool_fields = _mapping(info.get("rollout_extra_fields"))

    sample = _mapping(tool_fields.get("sample_info")) or _mapping(info.get("sample_info")) or info
    trace = tool_fields.get("tooleqa_trace", info.get("tooleqa_trace", [])) or []
    final_answer = tool_fields.get("tooleqa_final_answer", info.get("tooleqa_final_answer"))
    return sample, trace, final_answer, tool_fields or info


def score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
    **reward_kwargs: Any,
) -> dict[str, Any]:
    """Compute the scalar used by GRPO and preserve diagnostics for logging."""
    del data_source, solution_str
    sample, trace, final_answer, runtime_fields = _extract_runtime_fields(extra_info)
    if "answer" not in sample:
        sample = dict(sample)
        sample["answer"] = ground_truth
    judgment = None
    if sample.get("answer_setting") == "open":
        from src.train.RFT.open_protocol import semantic_judgment
        from src.train.RFT.resume_official import hardware_errors
        failures = hardware_errors(trace)
        if failures:
            raise RuntimeError("Hardware failure invalidates open rollout: " + failures[0])
        if final_answer is None:
            final_answer = extract_final_answer(trace)
        judgment = semantic_judgment(str(sample["question"]), str(sample["answer"]), str(final_answer or ""))
    result = compute_reward(sample, trace, final_answer=final_answer, weights=reward_kwargs,
                            semantic_score=judgment["score"] if judgment is not None else None)
    if judgment is not None:
        result["semantic_judgment"] = judgment
    attach_reward(
        runtime_fields.get("trajectory_id"),
        runtime_fields.get("global_step"),
        result,
        path_override=runtime_fields.get("trajectory_log_path"),
    )
    # VERL aggregates every returned key as a metric. Keep this interface flat
    # and numeric; the full trace remains in rollout extra_fields and can be
    # replayed offline with compute_reward for detailed audits.
    satisfied_facts = result["evidence"]["satisfied_facts"]
    metrics = {
        "score": result["score"],
        "evidence_objective": result["evidence_objective"],
        "answer_objective": result["answer_objective"],
        "informative_evidence": float(result["coverage"] > 0.0),
        "joint_reward_phase": float(result["reward_phase"] == "joint"),
        "acc": float(result["correct"]),
        "evidence_coverage": float(result["coverage"]),
        "evidence_complete": float(result["coverage"] >= 1.0),
        "grounded_fact_count": float(sum(fact.endswith(":grounded") for fact in satisfied_facts)),
        "position_fact_count": float(sum(fact.endswith(":position") for fact in satisfied_facts)),
        "size_fact_count": float(sum(fact.endswith(":size") for fact in satisfied_facts)),
        "visual_fact_count": float(sum(fact.endswith(":visual") for fact in satisfied_facts)),
        "task_fact_complete": float(any(fact.startswith("task:") for fact in satisfied_facts)),
        "invalid_count": float(result["evidence"]["invalid_count"]),
        "duplicate_count": float(result["evidence"]["duplicate_count"]),
        "no_progress_count": float(result["evidence"]["no_progress_count"]),
        "tool_count": float(result["evidence"]["tool_count"]),
        "tool_call_accuracy": result["tool_call_accuracy"],
        "tool_attempt_count": float(result["tool_attempt_count"]),
        "successful_tool_call_count": float(result["successful_tool_call_count"]),
        "final_answer_called": float(result["final_answer"] is not None),
        "forced_final": float(bool(runtime_fields.get("forced_final", False))),
        "truncated_normal_action": float(bool(runtime_fields.get("truncated_normal_action", False))),
    }
    metrics.update({f"reward_{name}": float(value) for name, value in result["components"].items()})
    metrics.update({name: float(value) for name, value in result["paper_metrics"].items()})
    if judgment is not None:
        metrics["llm_match"] = result["answer_quality"]
        metrics["semantic_score"] = float(judgment["score"])
    return metrics


compute_score = score
