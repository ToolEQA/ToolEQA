"""Evidence-grounded trajectory reward used by ToolEQA GRPO."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable, Mapping

from .evidence import EvidenceTracker, extract_final_answer, normalize_text
from .paper_metrics import compute_paper_metrics


@dataclass(frozen=True)
class RewardWeights:
    reward_phase: str = "joint"
    answer_correct: float = 1.0
    answer_wrong: float = -1.0
    # A missing answer must be worse than an unsupported wrong answer
    # (-1.0 - 0.5 - 0.3 = -1.8), otherwise GRPO learns to omit final_answer.
    answer_missing: float = -2.0
    # Evidence is the primary objective. Answer supervision is gated by the
    # achieved coverage below, so a correct unsupported guess has no answer
    # advantage over a wrong unsupported guess.
    evidence_terminal: float = 2.0
    premature_final: float = -0.5
    unsupported_final: float = -0.3
    invalid_action: float = -0.2
    duplicate_action: float = -0.1
    no_progress: float = -0.05
    tool_cost: float = -0.01
    path_excess: float = -0.2
    path_excess_cap: float = 2.0
    position_tolerance: float = 1.5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "RewardWeights":
        allowed = {item.name for item in fields(cls)}
        supplied = {key: value for key, value in (values or {}).items() if key in allowed and value is not None}
        return cls(**supplied)


def _normalize_answer(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"^(the )?(final )?answer (is )?", "", text).strip()
    return text


def _token_f1(prediction: str, target: str) -> float:
    pred_tokens = prediction.split()
    target_tokens = target.split()
    if not pred_tokens or not target_tokens:
        return 0.0
    common = 0
    remaining = list(target_tokens)
    for token in pred_tokens:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_is_correct(prediction: Any, sample: Mapping[str, Any]) -> bool:
    predicted = _normalize_answer(prediction)
    if not predicted:
        return False
    gold = _normalize_answer(sample.get("answer"))
    proposals = sample.get("proposals") or []
    canonical: set[str] = {gold} if gold else set()
    if len(gold) == 1 and gold in "abcd":
        index = ord(gold) - ord("a")
        if index < len(proposals):
            canonical.add(_normalize_answer(proposals[index]))
    elif gold and proposals:
        for index, proposal in enumerate(proposals[:4]):
            if _normalize_answer(proposal) == gold:
                canonical.add(chr(ord("a") + index))

    for target in canonical:
        if not target:
            continue
        if predicted == target:
            return True
        if target in {"yes", "no"} and predicted.split()[0] == target:
            return True
        if len(target) >= 3 and re.search(rf"\b{re.escape(target)}\b", predicted):
            return True
        if len(target.split()) > 1 and _token_f1(predicted, target) >= 0.9:
            return True
    return False


def compute_reward(
    sample: Mapping[str, Any],
    trace: Iterable[Mapping[str, Any]],
    final_answer: Any = None,
    weights: RewardWeights | Mapping[str, Any] | None = None,
    semantic_score: int | None = None,
) -> dict[str, Any]:
    """Return scalar reward plus an auditable component breakdown."""
    if isinstance(weights, RewardWeights):
        config = weights
    else:
        config = RewardWeights.from_mapping(weights)

    trace_list = list(trace or [])
    if final_answer is None:
        final_answer = extract_final_answer(trace_list)
    has_final = final_answer is not None and bool(str(final_answer).strip())
    is_open = sample.get("answer_setting") == "open"
    if is_open and (type(semantic_score) is not int or not 0 <= semantic_score <= 5):
        raise ValueError("Open answers require a validated semantic score in 0..5")
    quality = (semantic_score / 5.0 if has_final else 0.0) if is_open else None
    correct = (has_final and semantic_score == 5) if is_open else (has_final and answer_is_correct(final_answer, sample))

    tracker = EvidenceTracker(
        sample,
        position_tolerance=config.position_tolerance,
        require_audited_targets=True,
    )
    evidence = tracker.replay(trace_list)
    paper_metrics = compute_paper_metrics(sample, trace_list, correct=correct, answer_quality=quality)
    environment_actions = {"Navigate", "Location2D", "Location3D", "Crop", "VisualQA"}
    tool_attempts = [
        step
        for step in trace_list
        if step.get("action_type") in environment_actions
        or step.get("action_type") in {"InvalidCode", "Unknown"}
    ]
    successful_tool_calls = sum(
        step.get("action_type") in environment_actions
        and step.get("ok") is not False
        and not step.get("duplicate_rejected", False)
        for step in tool_attempts
    )
    tool_call_accuracy = successful_tool_calls / len(tool_attempts) if tool_attempts else 0.0

    phase = str(config.reward_phase).strip().lower()
    if phase not in {"evidence", "joint"}:
        raise ValueError(f"Unknown reward phase: {config.reward_phase!r}")

    if not has_final:
        answer_component = config.answer_missing
        answer_objective = config.answer_missing
    elif phase == "evidence":
        answer_component = 0.0
        answer_objective = 0.0
    else:
        answer_signal = 2 * quality - 1 if is_open else (config.answer_correct if correct else config.answer_wrong)
        answer_component = evidence.coverage * answer_signal
        answer_objective = answer_component

    components = {
        "answer": answer_component,
        "evidence_terminal": config.evidence_terminal * evidence.coverage,
        "premature_final": config.premature_final * (1.0 - evidence.coverage) if has_final else 0.0,
        "unsupported_final": config.unsupported_final if has_final and evidence.coverage <= 0.0 else 0.0,
        "invalid_action": config.invalid_action * evidence.invalid_count,
        "duplicate_action": config.duplicate_action * evidence.duplicate_count,
        "no_progress": config.no_progress * evidence.no_progress_count,
        "tool_cost": config.tool_cost * evidence.tool_count,
        "path_excess": 0.0,
    }

    expert_length = sample.get("traj_length", sample.get("expert_path_length"))
    try:
        expert_length = float(expert_length)
    except (TypeError, ValueError):
        expert_length = 0.0
    if evidence.path_length is not None and expert_length > 0:
        excess_ratio = max(evidence.path_length / expert_length - 1.0, 0.0)
        components["path_excess"] = config.path_excess * min(excess_ratio, config.path_excess_cap)

    # Efficiency costs are meaningful only after the trajectory has acquired
    # some evidence. Multiplying them by coverage prevents a zero-evidence
    # GRPO group from learning only to terminate cheaply.
    efficiency_names = (
        "invalid_action",
        "duplicate_action",
        "no_progress",
        "tool_cost",
        "path_excess",
    )
    efficiency_total = sum(components[name] for name in efficiency_names)
    evidence_objective = (
        components["evidence_terminal"]
        + components["premature_final"]
        + components["unsupported_final"]
        + evidence.coverage * efficiency_total
    )

    step_rewards: list[dict[str, Any]] = []
    for step in evidence.steps:
        # Positive evidence is deliberately paid once through terminal
        # sufficiency.  With a monotonic evidence state, summing delta here
        # would telescope to terminal coverage and double-count the same fact.
        value = 0.0
        value += config.invalid_action if step.invalid else 0.0
        value += config.duplicate_action if step.duplicate else 0.0
        value += config.no_progress if step.no_progress else 0.0
        if step.action not in {"Compute", "FinalAnswer", "InvalidCode", "Unknown"}:
            value += config.tool_cost
        step_rewards.append(
            {
                "step": step.index,
                "action": step.action,
                "reward": value,
                "delta": step.delta,
                "new_facts": list(step.new_facts),
            }
        )

    return {
        # ``score`` remains a fully auditable scalar containing every cost.
        # GDPO trains on the two explicit objectives below, so zero-evidence
        # groups can still receive zero learning signal even when their audit
        # scores expose invalid or inefficient behavior.
        "score": float(sum(components.values())),
        "reward_phase": phase,
        "evidence_objective": float(evidence_objective),
        "answer_objective": float(answer_objective),
        "correct": bool(correct),
        "answer_quality": float(quality if is_open else correct),
        "semantic_score": semantic_score,
        "final_answer": final_answer,
        "coverage": evidence.coverage,
        "components": components,
        "evidence": evidence.to_dict(),
        "paper_metrics": paper_metrics,
        # This is a protocol/execution diagnostic, not one of the paper's
        # success/Recall/EPath metrics and not a semantic tool-choice score.
        "tool_call_accuracy": float(tool_call_accuracy),
        "tool_attempt_count": int(len(tool_attempts)),
        "successful_tool_call_count": int(successful_tool_calls),
        "step_rewards": step_rewards,
        "weights": asdict(config),
    }
