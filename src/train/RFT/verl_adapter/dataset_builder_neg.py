"""
Negative sample construction for ToolEQA Offline RL.

For each positive sample, generates perturbed negative variants that teach the
policy what NOT to do.  GRPO's group-level advantage estimation naturally
assigns lower advantages to negative samples, achieving offline RL without
modifying verl internals.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional

NEG_TYPES = [
    "wrong_answer",
    "premature",
    "missing_tool",
    "redundant_nav",
    "wrong_object",
    "swapped_steps",
]

WRONG_OBJECT_DISTRACTORS = [
    "table", "chair", "lamp", "sofa", "desk", "shelf",
    "bed", "couch", "plant", "tv", "monitor", "cabinet",
    "rug", "curtain", "pillow", "blanket", "towel", "vase",
]


def _pick_wrong_answer(answer: str, proposals: List[str], rng: random.Random) -> Optional[str]:
    """Return a wrong answer different from the correct one."""
    if not proposals:
        return None
    options = [chr(65 + i) for i in range(len(proposals))]
    wrong = [o for o in options if o != answer]
    if not wrong:
        return None
    return rng.choice(wrong)


def _normalize_related_objects(objs: Any) -> List[str]:
    result = []
    for item in objs or []:
        if isinstance(item, dict):
            name = item.get("name")
            if name:
                result.append(str(name))
        elif item is not None:
            result.append(str(item))
    return result


def perturb_wrong_answer(record: Dict, rng: random.Random) -> Optional[Dict]:
    """Flip the ground-truth answer to a random wrong option."""
    extra = record.get("extra_info", {})
    answer = extra.get("answer")
    proposals = extra.get("proposals") or []
    if not answer or len(proposals) < 2:
        return None
    wrong = _pick_wrong_answer(answer, proposals, rng)
    if wrong is None:
        return None
    neg = copy.deepcopy(record)
    neg["extra_info"]["answer"] = wrong
    neg["extra_info"]["perturbation_type"] = "wrong_answer"
    neg["extra_info"]["is_negative"] = True
    neg["extra_info"]["neg_sample_id"] = f"{extra.get('sample_id', 'unknown')}_wrong_answer"
    return neg


def perturb_premature(record: Dict, rng: random.Random) -> Optional[Dict]:
    """Remove related_objects so the evidence check cannot pass."""
    extra = record.get("extra_info", {})
    related = extra.get("related_objects")
    if not related:
        return None
    neg = copy.deepcopy(record)
    neg["extra_info"]["related_objects"] = []
    neg["extra_info"]["perturbation_type"] = "premature"
    neg["extra_info"]["is_negative"] = True
    neg["extra_info"]["neg_sample_id"] = f"{extra.get('sample_id', 'unknown')}_premature"
    return neg


def perturb_missing_tool(record: Dict, rng: random.Random) -> Optional[Dict]:
    """Remove one key object from related_objects to simulate missing a critical tool call."""
    extra = record.get("extra_info", {})
    related = extra.get("related_objects") or []
    normalized = _normalize_related_objects(related)
    if len(normalized) < 2:
        return None
    neg = copy.deepcopy(record)
    objs = neg["extra_info"]["related_objects"]
    idx = rng.randint(0, len(objs) - 1)
    neg["extra_info"]["related_objects"] = objs[:idx] + objs[idx + 1:]
    neg["extra_info"]["perturbation_type"] = "missing_tool"
    neg["extra_info"]["is_negative"] = True
    neg["extra_info"]["neg_sample_id"] = f"{extra.get('sample_id', 'unknown')}_missing_tool"
    return neg


def perturb_redundant_nav(record: Dict, rng: random.Random) -> Optional[Dict]:
    """Mark the sample as having redundant navigation steps."""
    extra = record.get("extra_info", {})
    neg = copy.deepcopy(record)
    neg["extra_info"]["has_redundant_nav"] = True
    neg["extra_info"]["perturbation_type"] = "redundant_nav"
    neg["extra_info"]["is_negative"] = True
    neg["extra_info"]["neg_sample_id"] = f"{extra.get('sample_id', 'unknown')}_redundant_nav"
    return neg


def perturb_wrong_object(record: Dict, rng: random.Random) -> Optional[Dict]:
    """Replace a target object name with a plausible distractor."""
    extra = record.get("extra_info", {})
    related = extra.get("related_objects") or []
    normalized = _normalize_related_objects(related)
    if not normalized:
        return None
    neg = copy.deepcopy(record)
    objs = neg["extra_info"]["related_objects"]
    idx = rng.randint(0, len(objs) - 1)
    target = objs[idx]
    if isinstance(target, dict):
        current_name = target.get("name", "")
        distractors = [d for d in WRONG_OBJECT_DISTRACTORS if d != current_name]
        if distractors:
            target["name"] = rng.choice(distractors)
        else:
            return None
    else:
        distractors = [d for d in WRONG_OBJECT_DISTRACTORS if d != str(target)]
        if distractors:
            neg["extra_info"]["related_objects"][idx] = rng.choice(distractors)
        else:
            return None
    neg["extra_info"]["perturbation_type"] = "wrong_object"
    neg["extra_info"]["is_negative"] = True
    neg["extra_info"]["neg_sample_id"] = f"{extra.get('sample_id', 'unknown')}_wrong_object"
    return neg


def perturb_swapped_steps(record: Dict, rng: random.Random) -> Optional[Dict]:
    """Mark the sample as having step order swapped."""
    extra = record.get("extra_info", {})
    neg = copy.deepcopy(record)
    neg["extra_info"]["steps_swapped"] = True
    neg["extra_info"]["perturbation_type"] = "swapped_steps"
    neg["extra_info"]["is_negative"] = True
    neg["extra_info"]["neg_sample_id"] = f"{extra.get('sample_id', 'unknown')}_swapped_steps"
    return neg


PERTURB_FUNCTIONS = {
    "wrong_answer": perturb_wrong_answer,
    "premature": perturb_premature,
    "missing_tool": perturb_missing_tool,
    "redundant_nav": perturb_redundant_nav,
    "wrong_object": perturb_wrong_object,
    "swapped_steps": perturb_swapped_steps,
}


def generate_negatives(
    record: Dict,
    neg_types: Optional[List[str]] = None,
    neg_per_pos: int = 2,
    rng: Optional[random.Random] = None,
) -> List[Dict]:
    """
    Generate negative variants for a single positive record.

    Returns a list of perturbed records.  Some perturbation types may fail
    (e.g. only 1 proposal for wrong_answer), so the actual count may be
    less than ``neg_per_pos``.
    """
    if rng is None:
        rng = random.Random(42)
    if neg_types is None:
        neg_types = NEG_TYPES

    results = []
    chosen = rng.sample(neg_types, min(neg_per_pos, len(neg_types)))
    for neg_type in chosen:
        fn = PERTURB_FUNCTIONS[neg_type]
        neg = fn(record, rng)
        if neg is not None:
            results.append(neg)
    return results


def build_dataset_with_negatives(
    positive_records: List[Dict],
    neg_types: Optional[List[str]] = None,
    neg_per_pos: int = 2,
    seed: int = 42,
) -> List[Dict]:
    """
    Build a combined dataset of positive and negative samples.

    Returns all records (positive + negative) ready to be written as JSONL.
    """
    rng = random.Random(seed)
    all_records = list(positive_records)
    # Mark positive samples
    for rec in all_records:
        rec.setdefault("extra_info", {})
        rec["extra_info"]["is_negative"] = False
        rec["extra_info"]["perturbation_type"] = "none"

    for rec in positive_records:
        negs = generate_negatives(rec, neg_types=neg_types, neg_per_pos=neg_per_pos, rng=rng)
        all_records.extend(negs)

    return all_records
