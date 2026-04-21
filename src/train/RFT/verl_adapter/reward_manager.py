from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RewardBreakdown:
    r_ans: float = 0.0
    r_invalid: float = 0.0
    r_redund: float = 0.0
    r_prem: float = 0.0
    r_find: float = 0.0
    r_info: float = 0.0
    r_prog: float = 0.0
    r_hall: float = 0.0
    total: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class ToolEQARewardManager:
    """
    Function-based reward manager for ToolEQA controller RL.

    First implementation intentionally focuses on the most stable signals:
    answer correctness, invalid actions, redundant calls, and premature answer.
    """

    def __init__(
        self,
        answer_reward: float = 1.0,
        wrong_answer_reward: float = -1.0,
        invalid_penalty: float = -0.1,
        redundant_penalty: float = -0.05,
        premature_penalty: float = -0.3,
        find_reward: float = 0.2,
        info_reward: float = 0.1,
    ):
        self.answer_reward = answer_reward
        self.wrong_answer_reward = wrong_answer_reward
        self.invalid_penalty = invalid_penalty
        self.redundant_penalty = redundant_penalty
        self.premature_penalty = premature_penalty
        self.find_reward = find_reward
        self.info_reward = info_reward

    def _answer_is_correct(self, sample: Dict[str, Any], final_answer: Any) -> bool:
        gold = sample.get("answer")
        proposals = sample.get("proposals") or []
        if final_answer is None:
            return False
        final_answer = str(final_answer).strip()
        if gold is None:
            return False
        if final_answer == str(gold).strip():
            return True
        if proposals and gold in ["A", "B", "C", "D"]:
            idx = ord(gold) - ord("A")
            if 0 <= idx < len(proposals):
                return final_answer.lower() == str(proposals[idx]).strip().lower()
        return False

    def _freeze(self, value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(sorted((k, self._freeze(v)) for k, v in value.items()))
        if isinstance(value, list):
            return tuple(self._freeze(v) for v in value)
        return value

    def _normalize_related_objects(self, related_objects: Any) -> List[str]:
        normalized = []
        for item in related_objects or []:
            if isinstance(item, dict):
                name = item.get("name")
                if name:
                    normalized.append(str(name))
            elif item is not None:
                normalized.append(str(item))
        return normalized

    def compute(self, sample: Dict[str, Any], trace: List[Dict[str, Any]], final_answer: Any) -> RewardBreakdown:
        breakdown = RewardBreakdown()
        seen_actions = set()
        found_objects = set()

        for step in trace:
            signature = (step["action_type"], self._freeze(step["args"]))
            if signature in seen_actions:
                breakdown.r_redund += self.redundant_penalty
            else:
                seen_actions.add(signature)

            if step["action_type"] in {"ObjectLocation2D", "ObjectLocation3D"}:
                obj_name = step["args"].get("object")
                if obj_name and obj_name not in found_objects:
                    found_objects.add(obj_name)
                    if obj_name in self._normalize_related_objects(sample.get("related_objects")):
                        breakdown.r_find += self.find_reward

            if step["action_type"] in {"ObjectCrop", "VisualQA"}:
                breakdown.r_info += self.info_reward

        if self._answer_is_correct(sample, final_answer):
            breakdown.r_ans = self.answer_reward
        elif final_answer is not None:
            breakdown.r_ans = self.wrong_answer_reward

        required_objects = set(self._normalize_related_objects(sample.get("related_objects")))
        if final_answer is not None and required_objects and not required_objects.issubset(found_objects):
            breakdown.r_prem += self.premature_penalty

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
        breakdown.details = {
            "found_objects": sorted(found_objects),
            "required_objects": sorted(required_objects),
            "trace_length": len(trace),
        }
        return breakdown

    def __call__(self, sample: Dict[str, Any], trace: List[Dict[str, Any]], final_answer: Any) -> Dict[str, Any]:
        breakdown = self.compute(sample, trace, final_answer)
        return {
            "score": breakdown.total,
            "breakdown": breakdown.__dict__,
        }
