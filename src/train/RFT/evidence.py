"""Deterministic evidence-state tracking for ToolEQA rollouts.

The tracker deliberately rewards verified state transitions rather than raw tool
calls.  It is dependency-light so the same implementation can be unit-tested,
used by the VERL reward worker, and reused by offline evaluation scripts.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence


_ERROR_MARKERS = (
    "error",
    "failed",
    "not found",
    "no object",
    "invalid",
    "exception",
    "traceback",
    "cannot determine",
    "can't determine",
    "unable to determine",
    "not visible",
    "cannot see",
)

_OBJECT_ALIAS_GROUPS = (
    # HM3D commonly labels sofas/recliners with the broader ``chair`` class.
    {"chair", "armchair", "recliner", "sofa", "couch"},
    {"cabinet", "closet", "cupboard", "wardrobe"},
    {"clothes", "clothing", "hanger", "hangers", "shoe", "shoes"},
    {"rug", "mat", "doormat"},
    {"tap", "faucet"},
    {"picture", "frame", "painting", "artwork"},
    {"refrigerator", "fridge"},
    {"television", "tv"},
    {"ventilation", "hood", "range hood", "ventilation hood"},
    {"table", "island", "kitchen island"},
)

_GENERIC_OBJECT_ALIASES = {
    "appliance": {
        "refrigerator",
        "fridge",
        "dishwasher",
        "oven",
        "washer",
        "washing machine",
    },
}


def to_jsonable(value: Any) -> Any:
    """Convert common tensor/array objects to JSON-safe Python containers."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        try:
            return to_jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def normalize_text(value: Any) -> str:
    text = str(value or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_object_name(value: Any) -> str:
    text = normalize_text(value)
    tokens = text.split()
    while tokens and tokens[0] in {"a", "an", "the", "some"}:
        tokens.pop(0)
    return " ".join(tokens)


def _singular_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for token in normalize_object_name(value).split():
        tokens.add(token)
        if token.endswith("ies") and len(token) > 3:
            tokens.add(token[:-3] + "y")
        elif token.endswith("es") and len(token) > 3:
            tokens.add(token[:-2])
        elif token.endswith("s") and len(token) > 2:
            tokens.add(token[:-1])
    return tokens


def _alias_groups(value: Any) -> set[int]:
    normalized = normalize_object_name(value)
    tokens = _singular_tokens(normalized)
    matched: set[int] = set()
    for index, group in enumerate(_OBJECT_ALIAS_GROUPS):
        if any(alias in normalized or bool(_singular_tokens(alias) & tokens) for alias in group):
            matched.add(index)
    return matched


def object_names_match(left: Any, right: Any) -> bool:
    left_norm = normalize_object_name(left)
    right_norm = normalize_object_name(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if min(len(left_norm), len(right_norm)) >= 4 and (left_norm in right_norm or right_norm in left_norm):
        return True
    left_tokens = _singular_tokens(left_norm)
    right_tokens = _singular_tokens(right_norm)
    if left_tokens <= right_tokens or right_tokens <= left_tokens:
        return True
    for generic, specifics in _GENERIC_OBJECT_ALIASES.items():
        if (left_norm == generic and right_norm in specifics) or (
            right_norm == generic and left_norm in specifics
        ):
            return True
    return bool(_alias_groups(left_norm) & _alias_groups(right_norm))


def object_name_is_mentioned(name: Any, text: Any, *, max_ngram: int = 4) -> bool:
    """Return whether an annotated object has a defensible mention in text.

    EQA-RT inherits HM3D semantic labels, which are sometimes coarser than the
    noun used by the question (for example ``chair`` for a sofa or ``table``
    for a kitchen island).  Checking aliases over short question n-grams keeps
    those useful samples while rejecting unrelated labels that would make an
    otherwise sensible rollout receive zero evidence credit.
    """
    normalized_name = normalize_object_name(name)
    tokens = normalize_text(text).split()
    if not normalized_name or not tokens:
        return False
    for width in range(1, min(max_ngram, len(tokens)) + 1):
        for start in range(len(tokens) - width + 1):
            if object_names_match(normalized_name, " ".join(tokens[start : start + width])):
                return True
    return False


def unmentioned_related_objects(record: Mapping[str, Any]) -> list[str]:
    """List related-object labels that cannot be grounded in the question."""
    info = record.get("extra_info") or record
    question = info.get("question", record.get("question", ""))
    related = info.get("related_objects", record.get("related_objects", [])) or []
    missing: list[str] = []
    for item in related:
        name = item.get("name", "") if isinstance(item, Mapping) else item
        if not object_name_is_mentioned(name, question):
            missing.append(str(name))
    return missing


_CONTEXT_LINKERS = (
    ("across", "from"),
    ("adjacent", "to"),
    ("next", "to"),
    ("surrounded", "by"),
    ("positioned",),
    ("located",),
    ("beside",),
    ("under",),
    ("above",),
    ("below",),
    ("against",),
    ("facing",),
    ("near",),
    ("with",),
    ("on",),
    ("by",),
)


def _find_token_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> int | None:
    if not phrase or len(phrase) > len(tokens):
        return None
    width = len(phrase)
    for start in range(len(tokens) - width + 1):
        if tuple(tokens[start : start + width]) == tuple(phrase):
            return start
    return None


def _target_alias_phrases(name: Any) -> set[str]:
    """Return only defensible full-name aliases for role-level matching.

    ``object_names_match`` is intentionally permissive for open-vocabulary
    detector outputs. Question-role auditing must be stricter: a target named
    ``shower wall`` must not match the single word ``shower`` in ``shower door
    frame`` merely because the phrases share one token.
    """
    normalized = normalize_object_name(name)
    aliases = {normalized} if normalized else set()
    for group in _OBJECT_ALIAS_GROUPS:
        if normalized in group:
            aliases.update(group)
    if normalized in _GENERIC_OBJECT_ALIASES:
        aliases.update(_GENERIC_OBJECT_ALIASES[normalized])
    for generic, specifics in _GENERIC_OBJECT_ALIASES.items():
        if normalized in specifics:
            aliases.add(generic)
            aliases.update(specifics)
    return {normalize_object_name(alias) for alias in aliases if alias}


def _primary_target_in_clause(name: Any, clause: str) -> bool:
    """Whether ``name`` occurs as the head object rather than context.

    Generated EQA-RT descriptions often mention support objects after linkers,
    e.g. ``lamp on the table``. Such a mention must not make ``table`` pass as
    the compared operand. The canonical target is expected before the first
    contextual linker in its comparison clause.
    """
    tokens = normalize_text(clause).split()
    if not tokens:
        return False
    starts = []
    for alias in _target_alias_phrases(name):
        start = _find_token_phrase(tokens, alias.split())
        if start is not None:
            starts.append(start)
    if not starts:
        return False
    target_start = min(starts)
    linker_starts = [
        start
        for phrase in _CONTEXT_LINKERS
        if (start := _find_token_phrase(tokens, phrase)) is not None
    ]
    return not linker_starts or target_start <= min(linker_starts)


def _pairwise_comparison_clauses(question: Any) -> tuple[str, str] | None:
    text = normalize_text(question)
    for separator in (" or ", " than "):
        if separator in text:
            left, right = text.rsplit(separator, 1)
            if left.strip() and right.strip():
                return left.strip(), right.strip()
    return None


def _duplicate_choice_issues(info: Mapping[str, Any]) -> list[str]:
    proposals = info.get("proposals") or []
    if not proposals:
        return []
    normalized = [normalize_text(choice) for choice in proposals]
    issues: list[str] = []
    if len(normalized) != 4:
        issues.append(f"choice-count:{len(normalized)}")
    if len(set(normalized)) != len(normalized):
        issues.append("duplicate-choices")
    answer = str(info.get("answer", "")).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        issues.append(f"invalid-answer:{answer or 'missing'}")
    return issues


def evidence_target_issues(record: Mapping[str, Any]) -> list[str]:
    """Return annotation problems that make evidence reward misleading."""
    info = record.get("extra_info") or record
    issues = _duplicate_choice_issues(info)
    issues.extend(f"unmentioned:{name}" for name in unmentioned_related_objects(record))
    question_type = normalize_text(info.get("question_type", record.get("question_type", "")))
    related = info.get("related_objects", record.get("related_objects", [])) or []
    minimum_targets = 2 if question_type in {"attribute size", "attribute color", "distance distance"} else 1
    if len(related) < minimum_targets:
        issues.append(f"target-count:{len(related)}<{minimum_targets}")

    target_keys: list[tuple[str, str]] = []
    for index, item in enumerate(related):
        if not isinstance(item, Mapping):
            issues.append(f"malformed-target:{index}")
            continue
        target_keys.append((normalize_object_name(item.get("name")), str(item.get("id", index))))
    if len(set(target_keys)) != len(target_keys):
        issues.append("duplicate-targets")

    # Attribute comparisons in the source generator have two ordered operands. A
    # later natural-language enrichment pass occasionally replaced an operand
    # with one of its contextual objects while leaving choices/locations
    # untouched. Match each canonical target to the head of its own clause so
    # ``lamp on the table`` cannot validate ``table`` as the second operand.
    if question_type in {"attribute size", "attribute color"} and len(related) == 2:
        clauses = _pairwise_comparison_clauses(info.get("question", ""))
        if clauses is None:
            issues.append("unparsed-pairwise-question")
        else:
            for index, (target, clause) in enumerate(zip(related, clauses, strict=True)):
                if isinstance(target, Mapping) and not _primary_target_in_clause(target.get("name"), clause):
                    issues.append(f"nonprimary-target:{index}:{target.get('name', '')}")

    if question_type in {"attribute size", "distance distance"}:
        for left_index, left in enumerate(related):
            if not isinstance(left, Mapping):
                continue
            left_position = _finite_vector(left.get("pos"))
            if left_position is None:
                continue
            for right in related[left_index + 1 :]:
                if not isinstance(right, Mapping):
                    continue
                right_position = _finite_vector(right.get("pos"))
                if right_position is not None and _distance(left_position, right_position) <= 0.02:
                    issues.append(
                        f"coincident-pair:{left.get('name', '')}#{left.get('id', '')}:"
                        f"{right.get('name', '')}#{right.get('id', '')}"
                    )
    return list(dict.fromkeys(issues))


def build_evidence_target_audit(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build explicit reward-side targets without mutating source EQA-RT.

    The returned fields belong to a derived RFT manifest. Ineligible records
    retain candidate targets for diagnosis but must never reach reward
    computation.
    """
    info = record.get("extra_info") or record
    related = info.get("related_objects", record.get("related_objects", [])) or []
    targets = [to_jsonable(item) for item in related if isinstance(item, Mapping)]
    issues = evidence_target_issues(record)
    return {
        "evidence_targets": targets,
        "reward_eligible": not issues,
        "reward_audit": issues,
    }


def resolve_evidence_targets(
    sample: Mapping[str, Any], *, require_explicit: bool = False
) -> list[Mapping[str, Any]]:
    """Return audited targets, rejecting explicit ineligible manifests."""
    if "reward_eligible" in sample:
        issues = list(sample.get("reward_audit") or [])
        if sample.get("reward_eligible") is not True or issues:
            detail = ", ".join(str(issue) for issue in issues) or "not reward eligible"
            raise ValueError(f"Cannot compute evidence reward for this sample: {detail}")
        targets = sample.get("evidence_targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError("Reward-eligible sample has no evidence_targets")
        resolved = [item for item in targets if isinstance(item, Mapping)]
        if len(resolved) != len(targets):
            raise ValueError("Reward-eligible sample has malformed evidence_targets")
        if "related_objects" in sample:
            source_targets = [
                to_jsonable(item)
                for item in (sample.get("related_objects") or [])
                if isinstance(item, Mapping)
            ]
            if to_jsonable(resolved) != source_targets:
                raise ValueError("evidence_targets do not match audited related_objects")
        return resolved

    if require_explicit:
        raise ValueError(
            "Cannot compute evidence reward from an unaudited sample: "
            "reward_eligible/evidence_targets are required"
        )

    # Backward-compatible path for focused tracker tests and offline audit
    # tools. Reward computation always sets ``require_explicit=True``.
    return [item for item in (sample.get("related_objects") or []) if isinstance(item, Mapping)]


def normalize_action(value: Any) -> str:
    compact = re.sub(r"[^a-z0-9]", "", str(value or "").lower())
    aliases = {
        "gonextpoint": "Navigate",
        "gonextpointtool": "Navigate",
        "navigate": "Navigate",
        "objectlocation2d": "Location2D",
        "objectlocation2dtool": "Location2D",
        "location2d": "Location2D",
        "objectlocation3d": "Location3D",
        "objectlocation3dtool": "Location3D",
        "location3d": "Location3D",
        "objectcrop": "Crop",
        "objectcroptool": "Crop",
        "crop": "Crop",
        "visualqa": "VisualQA",
        "visualqatool": "VisualQA",
        "vqa": "VisualQA",
        "finalanswer": "FinalAnswer",
        "finalanswertool": "FinalAnswer",
        "compute": "Compute",
        "computeaction": "Compute",
        "invalidcode": "InvalidCode",
    }
    return aliases.get(compact, str(value or "Unknown"))


def _finite_vector(value: Any, min_size: int = 3) -> list[float] | None:
    value = to_jsonable(value)
    if not isinstance(value, list) or len(value) < min_size:
        return None
    try:
        vector = [float(number) for number in value[:min_size]]
    except (TypeError, ValueError):
        return None
    return vector if all(math.isfinite(number) for number in vector) else None


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _contains_error(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        lowered = value.lower()
        return not lowered.strip() or any(marker in lowered for marker in _ERROR_MARKERS)
    if isinstance(value, Mapping):
        if value.get("ok") is False or value.get("error"):
            return True
    return False


def _extract_query(args: Any) -> str:
    if isinstance(args, Mapping):
        for key in ("object", "object_name", "query", "name", "target"):
            if key in args:
                return normalize_object_name(args[key])
        if args:
            return normalize_object_name(next(iter(args.values())))
    if isinstance(args, (list, tuple)) and args:
        return normalize_object_name(args[0])
    return normalize_object_name(args)


def _extract_vqa_question(args: Any) -> str:
    if isinstance(args, Mapping):
        for key in ("question", "query", "text"):
            if key in args:
                return normalize_text(args[key])
        values = list(args.values())
        if values:
            return normalize_text(values[-1])
    if isinstance(args, (list, tuple)) and args:
        return normalize_text(args[-1])
    return normalize_text(args)


def _extract_boxes(result: Any) -> list[list[float]]:
    result = to_jsonable(result)
    if isinstance(result, Mapping):
        for key in ("bboxes_2d", "bboxes", "boxes", "bbox", "detections"):
            if key in result:
                result = result[key]
                break
    if isinstance(result, list) and result and not isinstance(result[0], (list, tuple, Mapping)):
        result = [result]
    boxes: list[list[float]] = []
    if not isinstance(result, list):
        return boxes
    for item in result:
        if isinstance(item, Mapping):
            item = item.get("bbox", item.get("box", []))
        vector = _finite_vector(item, min_size=4)
        if vector and vector[2] > vector[0] and vector[3] > vector[1]:
            boxes.append(vector[:4])
    return boxes


def _extract_scores(result: Any) -> list[float]:
    result = to_jsonable(result)
    if not isinstance(result, Mapping):
        return []
    values = result.get("scores", result.get("confidences", []))
    if not isinstance(values, list):
        values = [values]
    scores: list[float] = []
    for value in values:
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            scores.append(score)
    return scores


def _image_paths(step: Mapping[str, Any]) -> set[str]:
    values: list[Any] = []
    resolved = step.get("resolved_image_paths")
    if isinstance(resolved, (list, tuple)):
        values.extend(resolved)
    elif resolved:
        values.append(resolved)
    args = step.get("args")
    if isinstance(args, Mapping):
        for key in ("image_path", "image_paths"):
            value = args.get(key)
            if isinstance(value, (list, tuple)):
                values.extend(value)
            elif value:
                values.append(value)
    if not values and step.get("image_path_before"):
        values.append(step["image_path_before"])
    return {os.path.realpath(os.path.abspath(os.fspath(value))) for value in values if value}


def _result_image_paths(result: Any) -> set[str]:
    result = to_jsonable(result)
    if isinstance(result, Mapping):
        for key in ("image_paths", "paths", "path", "output_paths"):
            if key in result:
                result = result[key]
                break
    if isinstance(result, (str, os.PathLike)):
        result = [result]
    if not isinstance(result, list):
        return set()
    return {
        os.path.realpath(os.path.abspath(os.fspath(value)))
        for value in result
        if isinstance(value, (str, os.PathLike)) and value
    }


def _extract_centers_and_sizes(result: Any) -> tuple[list[list[float]], list[list[float]]]:
    result = to_jsonable(result)
    centers: Any = None
    sizes: Any = None
    if isinstance(result, Mapping):
        centers = result.get("centers", result.get("center"))
        sizes = result.get("sizes", result.get("size"))
        if centers is None and "bboxes_3d" in result:
            rows = result["bboxes_3d"]
            centers = [row[:3] for row in rows]
            sizes = [row[3:6] for row in rows]
    elif isinstance(result, list) and len(result) >= 2:
        centers, sizes = result[0], result[1]

    if centers is not None and (not isinstance(centers, list) or not centers or not isinstance(centers[0], list)):
        centers = [centers]
    if sizes is not None and (not isinstance(sizes, list) or not sizes or not isinstance(sizes[0], list)):
        sizes = [sizes]

    valid_centers = [vector for item in (centers or []) if (vector := _finite_vector(item))]
    valid_sizes = [vector for item in (sizes or []) if (vector := _finite_vector(item))]
    return valid_centers, valid_sizes


@dataclass(frozen=True)
class EvidenceTarget:
    key: str
    name: str
    object_id: str
    position: tuple[float, float, float] | None


@dataclass(frozen=True)
class EvidenceSpec:
    question_type: str
    evidence_kind: str
    target_requirements: tuple[str, ...]
    task_fact: str | None
    targets: tuple[EvidenceTarget, ...]

    @classmethod
    def from_sample(
        cls, sample: Mapping[str, Any], *, require_audited_targets: bool = False
    ) -> "EvidenceSpec":
        question_type = normalize_text(sample.get("question_type"))
        if question_type == "attribute size":
            evidence_kind = "geometry_comparison"
            target_requirements = ("grounded", "position", "size")
            task_fact = "task:geometry_comparison"
        elif question_type.startswith("distance"):
            evidence_kind = "distance_relation"
            target_requirements = ("grounded", "position")
            task_fact = "task:distance_relation"
        elif question_type.startswith("count"):
            evidence_kind = "instance_set"
            target_requirements = ("grounded", "position")
            task_fact = "task:instance_count"
        elif question_type == "location location":
            # A 3D center identifies the instance, while a grounded visual
            # observation supplies the semantic room label that coordinates
            # alone cannot express.
            evidence_kind = "room_localization"
            target_requirements = ("grounded", "position", "visual")
            task_fact = "task:room_localization"
        elif question_type == "location special":
            evidence_kind = "relative_localization"
            target_requirements = ("grounded", "position", "visual")
            task_fact = "task:relative_localization"
        elif question_type == "attribute color":
            evidence_kind = "visual_comparison"
            target_requirements = ("grounded", "visual")
            task_fact = "task:visual_comparison"
        elif question_type == "attribute special":
            evidence_kind = "visual_attribute"
            target_requirements = ("grounded", "visual")
            task_fact = "task:attribute_resolution"
        elif question_type.startswith("relationship"):
            # EQA-RT relationship questions ask about contextual scene facts
            # around one grounded target, not metric pairwise geometry.
            evidence_kind = "scene_relationship"
            target_requirements = ("grounded", "visual")
            task_fact = "task:scene_relationship"
        elif question_type.startswith("status"):
            evidence_kind = "visual_status"
            target_requirements = ("grounded", "visual")
            task_fact = "task:status_resolution"
        else:
            evidence_kind = "grounded"
            target_requirements = ("grounded",)
            task_fact = None

        targets: list[EvidenceTarget] = []
        for index, obj in enumerate(
            resolve_evidence_targets(sample, require_explicit=require_audited_targets)
        ):
            if not isinstance(obj, Mapping):
                continue
            name = normalize_object_name(obj.get("name"))
            object_id = str(obj.get("id", index))
            pos = _finite_vector(obj.get("pos"))
            key = f"{name or 'object'}#{object_id}"
            targets.append(
                EvidenceTarget(
                    key=key,
                    name=name,
                    object_id=object_id,
                    position=tuple(pos) if pos else None,
                )
            )
        return cls(
            question_type=question_type,
            evidence_kind=evidence_kind,
            target_requirements=target_requirements,
            task_fact=task_fact,
            targets=tuple(targets),
        )

    @property
    def required_facts(self) -> tuple[str, ...]:
        facts: list[str] = []
        for target in self.targets:
            facts.extend(f"{target.key}:{requirement}" for requirement in self.target_requirements)
        if self.targets and self.task_fact:
            facts.append(self.task_fact)
        return tuple(facts)


@dataclass
class EvidenceStep:
    index: int
    action: str
    delta: float
    coverage_after: float
    new_facts: list[str] = field(default_factory=list)
    invalid: bool = False
    duplicate: bool = False
    no_progress: bool = False


@dataclass
class EvidenceReport:
    question_type: str
    evidence_kind: str
    coverage: float
    satisfied_facts: list[str]
    required_facts: list[str]
    steps: list[EvidenceStep]
    invalid_count: int
    duplicate_count: int
    no_progress_count: int
    tool_count: int
    path_length: float | None
    measurements: dict[str, dict[str, list[float]]]
    derived_evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["progress_sum"] = sum(max(step.delta, 0.0) for step in self.steps)
        return result


class EvidenceTracker:
    """Replay a rollout trace into a monotonic, task-aware evidence state."""

    def __init__(
        self,
        sample: Mapping[str, Any],
        position_tolerance: float = 1.5,
        *,
        require_audited_targets: bool = False,
    ):
        self.sample = dict(sample)
        self.spec = EvidenceSpec.from_sample(
            sample, require_audited_targets=require_audited_targets
        )
        self.position_tolerance = float(position_tolerance)
        self.grounded: set[str] = set()
        self.positions: set[str] = set()
        self.sizes: set[str] = set()
        self.visual: set[str] = set()
        self.measurements: dict[str, dict[str, list[float]]] = {}
        self.grounded_images: dict[str, set[str]] = {}
        self._signatures: set[str] = set()

    def _satisfied_facts(self) -> set[str]:
        facts: set[str] = set()
        for target in self.spec.targets:
            if target.key in self.grounded:
                facts.add(f"{target.key}:grounded")
            if "position" in self.spec.target_requirements and target.key in self.positions:
                facts.add(f"{target.key}:position")
            if "size" in self.spec.target_requirements and target.key in self.sizes:
                facts.add(f"{target.key}:size")
            if "visual" in self.spec.target_requirements and target.key in self.visual:
                facts.add(f"{target.key}:visual")
        if self.spec.targets and self.spec.task_fact:
            target_facts = {
                f"{target.key}:{requirement}"
                for target in self.spec.targets
                for requirement in self.spec.target_requirements
            }
            if target_facts <= facts:
                facts.add(self.spec.task_fact)
        return facts

    def _coverage(self) -> float:
        required = self.spec.required_facts
        return len(self._satisfied_facts()) / len(required) if required else 0.0

    def _derived_evidence(self) -> dict[str, Any]:
        """Return auditable 3D operands derived only from verified tool output."""
        result: dict[str, Any] = {}
        ordered = [target for target in self.spec.targets if target.key in self.measurements]

        if self.spec.evidence_kind == "geometry_comparison":
            result["objects"] = [
                {
                    "key": target.key,
                    "size": self.measurements[target.key].get("size"),
                    "height": self.measurements[target.key].get("size", [None, None, None])[2],
                    "volume": math.prod(self.measurements[target.key]["size"])
                    if len(self.measurements[target.key].get("size", [])) == 3
                    else None,
                }
                for target in ordered
            ]
        elif self.spec.evidence_kind == "distance_relation":
            distances: list[dict[str, Any]] = []
            for left_index, left in enumerate(ordered):
                left_pos = self.measurements[left.key].get("position")
                if left_pos is None:
                    continue
                for right in ordered[left_index + 1 :]:
                    right_pos = self.measurements[right.key].get("position")
                    if right_pos is not None:
                        distances.append(
                            {
                                "left": left.key,
                                "right": right.key,
                                "distance": _distance(left_pos, right_pos),
                            }
                        )
            result["pairwise_distances"] = distances
        elif self.spec.evidence_kind == "instance_set":
            result["verified_instances"] = [
                {"key": target.key, "position": self.measurements[target.key].get("position")}
                for target in ordered
                if target.key in self.positions
            ]
            result["verified_count"] = len(result["verified_instances"])
        elif self.spec.evidence_kind in {"room_localization", "relative_localization"}:
            result["localized_objects"] = [
                {"key": target.key, "position": self.measurements[target.key].get("position")}
                for target in ordered
                if target.key in self.positions
            ]
        return result

    def _matching_targets(self, query: str) -> list[EvidenceTarget]:
        if not query:
            return []
        return [target for target in self.spec.targets if object_names_match(query, target.name)]

    def _apply_2d(self, args: Any, result: Any, images: set[str]) -> bool:
        boxes = _extract_boxes(result)
        if not boxes:
            return False
        scores = _extract_scores(result)
        if scores and max(scores) < 0.37:
            return False
        query = _extract_query(args)
        json_result = to_jsonable(result)
        if isinstance(json_result, Mapping):
            labels = json_result.get("labels") or []
            if labels and not any(object_names_match(query, label) for label in labels):
                return False
        candidates = [target for target in self._matching_targets(query) if target.key not in self.grounded]
        for target in candidates[: len(boxes)]:
            self.grounded.add(target.key)
            self.grounded_images.setdefault(target.key, set()).update(images)
        return True

    def _apply_3d(self, args: Any, result: Any, images: set[str]) -> bool:
        centers, sizes = _extract_centers_and_sizes(result)
        if not centers or not sizes or not any(all(side > 0 for side in size) for size in sizes):
            return False
        candidates = self._matching_targets(_extract_query(args))
        if not candidates:
            return False

        # A valid class-conditioned 3D detection establishes visual grounding.
        # Geometry is a stronger, instance-level fact and additionally requires
        # agreement with the annotated Habitat-world center below.  Keeping the
        # two transitions separate avoids turning ordinary detector localization
        # error into "no evidence at all" while still rejecting a wrong instance
        # for geometry-dependent questions.
        for target in candidates[: len(centers)]:
            self.grounded.add(target.key)
            self.grounded_images.setdefault(target.key, set()).update(images)

        unused_centers = set(range(len(centers)))
        for target in candidates:
            if target.position is None:
                best_index = next(iter(unused_centers), None)
            else:
                distances = [(index, _distance(centers[index], target.position)) for index in unused_centers]
                best_index, best_distance = min(distances, key=lambda item: item[1], default=(None, math.inf))
                if best_distance > self.position_tolerance:
                    best_index = None
            if best_index is not None:
                unused_centers.discard(best_index)
                self.positions.add(target.key)
                measurement = self.measurements.setdefault(target.key, {})
                measurement["position"] = centers[best_index]
                if best_index < len(sizes) and all(side > 0 for side in sizes[best_index]):
                    self.sizes.add(target.key)
                    measurement["size"] = sizes[best_index]
        return True

    def _apply_vqa(self, args: Any, result: Any, images: set[str]) -> bool:
        if _contains_error(result):
            return False
        question = _extract_vqa_question(args)
        candidates = [
            target
            for target in self.spec.targets
            if target.key in self.grounded
            and target.name
            and object_names_match(target.name, question)
            and images
            and bool(images & self.grounded_images.get(target.key, set()))
        ]
        if not candidates:
            source_question = normalize_text(self.sample.get("question"))
            source_tokens = set(source_question.split())
            query_tokens = set(question.split())
            overlap = len(source_tokens & query_tokens) / max(len(source_tokens), 1)
            if overlap >= 0.5:
                candidates = [
                    target
                    for target in self.spec.targets
                    if target.key in self.grounded
                    and images
                    and bool(images & self.grounded_images.get(target.key, set()))
                ]
        for target in candidates:
            self.visual.add(target.key)
        return bool(candidates)

    def _apply_crop(self, result: Any, source_images: set[str]) -> bool:
        """Propagate grounding provenance from a source view to its crops."""
        crop_images = _result_image_paths(result)
        if not crop_images or not source_images:
            return False
        matched = False
        for target_key, grounded in self.grounded_images.items():
            if grounded & source_images:
                grounded.update(crop_images)
                matched = True
        return matched

    @staticmethod
    def _signature(action: str, args: Any, image_context: Any) -> str:
        payload = {
            "action": action,
            "args": to_jsonable(args),
            "image": str(image_context or ""),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    def replay(self, trace: Iterable[Mapping[str, Any]]) -> EvidenceReport:
        steps: list[EvidenceStep] = []
        path_length: float | None = None
        evidence_actions = {"Location2D", "Location3D", "VisualQA"}

        for index, raw_step in enumerate(trace or []):
            step = dict(raw_step)
            action = normalize_action(step.get("action_type", step.get("tool", step.get("action"))))
            before_facts = self._satisfied_facts()
            before_coverage = self._coverage()
            explicit_error = step.get("ok") is False or bool(step.get("error"))
            invalid = explicit_error or action == "InvalidCode"

            signature = self._signature(action, step.get("args"), step.get("image_path_before"))
            duplicate = bool(step.get("duplicate_rejected")) or (
                action != "FinalAnswer" and signature in self._signatures
            )
            if action != "FinalAnswer":
                self._signatures.add(signature)

            if not invalid:
                result = step.get("result")
                images = _image_paths(step)
                if action == "Location2D":
                    boxes = _extract_boxes(result)
                    # An empty detector result is a valid observation with no
                    # evidence progress. Malformed/low-confidence detections
                    # remain invalid and keep their stronger penalty.
                    invalid = bool(boxes) and not self._apply_2d(step.get("args"), result, images)
                elif action == "Location3D":
                    coordinate_frame = step.get("coordinate_frame")
                    centers, sizes = _extract_centers_and_sizes(result)
                    empty_detection = not centers and not sizes
                    invalid = coordinate_frame not in {None, "habitat_world"} or (
                        not empty_detection and not self._apply_3d(step.get("args"), result, images)
                    )
                elif action == "VisualQA":
                    invalid = _contains_error(result)
                    if not invalid:
                        self._apply_vqa(step.get("args"), result, images)
                elif action == "Crop":
                    invalid = _contains_error(result) or not _result_image_paths(result)
                    if not invalid:
                        self._apply_crop(result, images)

            after_facts = self._satisfied_facts()
            coverage = self._coverage()
            delta = max(coverage - before_coverage, 0.0)
            no_progress = action in evidence_actions and not invalid and delta <= 0.0

            candidate_path = step.get("path_length_after", step.get("path_length"))
            try:
                if candidate_path is not None and math.isfinite(float(candidate_path)):
                    path_length = max(path_length or 0.0, float(candidate_path))
            except (TypeError, ValueError):
                pass

            steps.append(
                EvidenceStep(
                    index=int(step.get("step", index)),
                    action=action,
                    delta=delta,
                    coverage_after=coverage,
                    new_facts=sorted(after_facts - before_facts),
                    invalid=invalid,
                    duplicate=duplicate,
                    no_progress=no_progress,
                )
            )

        return EvidenceReport(
            question_type=self.spec.question_type,
            evidence_kind=self.spec.evidence_kind,
            coverage=self._coverage(),
            satisfied_facts=sorted(self._satisfied_facts()),
            required_facts=list(self.spec.required_facts),
            steps=steps,
            invalid_count=sum(step.invalid for step in steps),
            duplicate_count=sum(step.duplicate for step in steps),
            no_progress_count=sum(step.no_progress for step in steps),
            tool_count=sum(
                step.action not in {"Compute", "FinalAnswer", "InvalidCode", "Unknown"}
                for step in steps
            ),
            path_length=path_length,
            measurements=to_jsonable(self.measurements),
            derived_evidence=to_jsonable(self._derived_evidence()),
        )


def extract_final_answer(trace: Iterable[Mapping[str, Any]]) -> str | None:
    for step in reversed(list(trace or [])):
        if normalize_action(step.get("action_type", step.get("tool", step.get("action")))) != "FinalAnswer":
            continue
        args = step.get("args")
        if isinstance(args, Mapping):
            for key in ("answer", "final_answer", "response"):
                if key in args:
                    return str(args[key])
            if args:
                return str(next(iter(args.values())))
        if isinstance(args, (list, tuple)) and args:
            return str(args[0])
        if args is not None:
            return str(args)
        if step.get("result") is not None:
            return str(step["result"])
    return None
