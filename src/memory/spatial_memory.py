from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple


class SpatialMemory:
    """Structured spatial evidence buffer with scene graph query layer.

    Maintains a flat object-wise buffer (matching the paper) where each detected
    object stores its 3D position, size, bounding box, crop paths, and the step
    at which it was detected. Scene graph relations are computed on-the-fly from
    the stored positions.

    Deterministic — no learnable parameters. Updated from tool outputs.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.explored_steps: int = 0
        self.detected_objects: Dict[str, Dict[str, Any]] = {}
        self.vqa_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Programmatic update (when structured tool output is available)
    # ------------------------------------------------------------------

    def update(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Any,
        step_idx: int,
    ) -> None:
        """Update buffer from structured tool output."""
        if tool_name == "GoNextPointTool":
            self.explored_steps += 1

        elif tool_name == "ObjectLocation3D":
            self._update_location_3d(tool_args, tool_result, step_idx)

        elif tool_name == "ObjectLocation2D":
            self._update_location_2d(tool_args, tool_result, step_idx)

        elif tool_name == "ObjectCrop":
            self._update_crop(tool_args, tool_result)

        elif tool_name in ("VisualQATool", "VisualQA"):
            self._update_vqa(tool_args, tool_result, step_idx)

    def _update_location_3d(
        self, args: Dict[str, Any], result: Any, step_idx: int
    ) -> None:
        obj_name = args.get("object", "unknown")
        center, size = None, None
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            center = self._extract_xyz(result[0])
            size = self._extract_xyz(result[1])
        if center is None:
            return
        entry = self._get_or_create_object(obj_name)
        entry["position"] = center
        entry["size"] = size
        entry["step"] = step_idx
        entry["image_path"] = args.get("image_path", "")

    def _update_location_2d(
        self, args: Dict[str, Any], result: Any, step_idx: int
    ) -> None:
        obj_name = args.get("object", "unknown")
        if isinstance(result, dict):
            bboxes = result.get("bboxes_2d", [])
            if bboxes:
                entry = self._get_or_create_object(obj_name)
                entry["bbox_2d"] = bboxes[0] if len(bboxes) == 1 else bboxes
                entry["step"] = step_idx
                entry["image_path"] = args.get("image_path", "")

    def _update_crop(self, args: Dict[str, Any], result: Any) -> None:
        obj_name = self._infer_crop_object(args)
        if obj_name is None:
            return
        entry = self._get_or_create_object(obj_name)
        paths = result if isinstance(result, list) else [result]
        existing = entry.get("crop_paths", [])
        for p in paths:
            if p and p not in existing:
                existing.append(p)
        entry["crop_paths"] = existing

    def _update_vqa(
        self, args: Dict[str, Any], result: Any, step_idx: int
    ) -> None:
        question = args.get("question", "")
        answer = str(result) if result is not None else ""
        self.vqa_results.append(
            {"question": question, "answer": answer, "step": step_idx}
        )

    # ------------------------------------------------------------------
    # Update from observation text (training data / fallback)
    # ------------------------------------------------------------------

    def update_from_observation(
        self,
        code_text: str,
        observation_text: str,
        step_idx: int,
    ) -> None:
        """Parse tool call from code and result from observation text."""
        code_lower = code_text.lower()

        # Detect GoNextPointTool calls
        if "gonextpointtool" in code_lower or "go_next_point" in code_lower:
            self.explored_steps += 1

        # Detect ObjectLocation3D
        if "objectlocation3d" in code_lower:
            obj = self._extract_tool_arg(code_text, "object")
            center, size = self._parse_3d_result(observation_text)
            if obj and center:
                entry = self._get_or_create_object(obj)
                entry["position"] = center
                entry["size"] = size
                entry["step"] = step_idx

        # Detect ObjectLocation2D
        if "objectlocation2d" in code_lower:
            obj = self._extract_tool_arg(code_text, "object")
            bboxes, labels = self._parse_2d_result(observation_text)
            if obj and bboxes:
                entry = self._get_or_create_object(obj)
                entry["bbox_2d"] = bboxes[0] if len(bboxes) == 1 else bboxes
                entry["step"] = step_idx
                if labels:
                    entry["label_2d"] = labels[0] if len(labels) == 1 else labels

        # Detect ObjectCrop
        if "objectcrop" in code_lower:
            paths = self._parse_crop_paths(observation_text)
            obj = self._infer_crop_object_from_code(code_text)
            if obj and paths:
                entry = self._get_or_create_object(obj)
                existing = entry.get("crop_paths", [])
                for p in paths:
                    if p and p not in existing:
                        existing.append(p)
                entry["crop_paths"] = existing

        # Detect VisualQA
        if "visualqatool" in code_lower or "visualqa" in code_lower:
            question = self._extract_tool_arg(code_text, "question")
            if question:
                self.vqa_results.append(
                    {"question": question, "answer": observation_text.strip(), "step": step_idx}
                )

    # ------------------------------------------------------------------
    # Scene graph: compute spatial relations on-the-fly
    # ------------------------------------------------------------------

    def compute_relations(
        self, distance_threshold: float = 1.5
    ) -> Dict[str, List[Tuple[str, str, Dict[str, float]]]]:
        """Compute spatial relations between detected objects.

        Returns:
            Dict mapping relation type to list of (obj_a, obj_b, metadata).
            Relation types: near, above, below, left, right, in_front, behind.
        """
        relations: Dict[str, List[Tuple[str, str, Dict[str, float]]]] = {
            "near": [],
            "above": [],
            "below": [],
            "left": [],
            "right": [],
            "in_front": [],
            "behind": [],
        }
        objects_with_pos = {
            name: info
            for name, info in self.detected_objects.items()
            if "position" in info and info["position"] is not None
        }
        names = list(objects_with_pos.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pos_a = objects_with_pos[a]["position"]
                pos_b = objects_with_pos[b]["position"]
                dx = pos_b[0] - pos_a[0]  # x: left-right
                dy = pos_b[1] - pos_a[1]  # y: up-down (height)
                dz = pos_b[2] - pos_a[2]  # z: forward-back
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)

                meta = {"distance": round(dist, 2)}

                if dist <= distance_threshold:
                    relations["near"].append((a, b, meta))

                if abs(dy) > 0.3:
                    if dy > 0:
                        relations["above"].append((b, a, {"dy": round(dy, 2)}))
                    else:
                        relations["below"].append((b, a, {"dy": round(dy, 2)}))

                if abs(dx) > 0.3:
                    if dx > 0:
                        relations["right"].append((b, a, {"dx": round(dx, 2)}))
                    else:
                        relations["left"].append((b, a, {"dx": round(dx, 2)}))

                if abs(dz) > 0.3:
                    if dz > 0:
                        relations["in_front"].append((b, a, {"dz": round(dz, 2)}))
                    else:
                        relations["behind"].append((b, a, {"dz": round(dz, 2)}))

        return relations

    def get_nearby_objects(
        self, obj_name: str, radius: float = 1.5
    ) -> List[Tuple[str, float]]:
        """Return objects within *radius* meters of *obj_name*."""
        if obj_name not in self.detected_objects:
            return []
        pos = self.detected_objects[obj_name].get("position")
        if pos is None:
            return []
        nearby = []
        for name, info in self.detected_objects.items():
            if name == obj_name:
                continue
            other_pos = info.get("position")
            if other_pos is None:
                continue
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, other_pos)))
            if dist <= radius:
                nearby.append((name, round(dist, 2)))
        nearby.sort(key=lambda x: x[1])
        return nearby

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> str:
        """Serialize as flat text (matching the paper format)."""
        if not self.explored_steps and not self.detected_objects and not self.vqa_results:
            return ""

        lines = ["[Spatial Memory]"]
        lines.append(f"Explored: {self.explored_steps} viewpoints")

        if self.detected_objects:
            lines.append("Detected objects:")
            for name, info in self.detected_objects.items():
                parts = [name]
                if "position" in info:
                    parts.append(f"position={info['position']}")
                if "size" in info:
                    parts.append(f"size={info['size']}")
                if "step" in info:
                    parts.append(f"step={info['step']}")
                if "crop_paths" in info and info["crop_paths"]:
                    parts.append(f"crop={info['crop_paths'][0]}")
                lines.append(f"  - {', '.join(parts)}")

        if self.vqa_results:
            lines.append("VQA results:")
            for vqa in self.vqa_results:
                q = vqa["question"][:80]
                a = vqa["answer"][:120]
                lines.append(f"  - step {vqa['step']}: Q=\"{q}\" A=\"{a}\"")

        return "\n".join(lines)

    def serialize_with_relations(self, distance_threshold: float = 1.5) -> str:
        """Serialize with spatial relations (scene graph format)."""
        base = self.serialize()
        if not base:
            return ""

        relations = self.compute_relations(distance_threshold)
        rel_lines = []
        for rel_type, pairs in relations.items():
            for a, b, meta in pairs:
                if rel_type == "near":
                    rel_lines.append(f"  - {a} is near {b} (distance={meta['distance']}m)")
                elif rel_type == "above":
                    rel_lines.append(f"  - {a} is above {b}")
                elif rel_type == "below":
                    rel_lines.append(f"  - {a} is below {b}")
                elif rel_type == "left":
                    rel_lines.append(f"  - {a} is left of {b}")
                elif rel_type == "right":
                    rel_lines.append(f"  - {a} is right of {b}")
                elif rel_type == "in_front":
                    rel_lines.append(f"  - {a} is in front of {b}")
                elif rel_type == "behind":
                    rel_lines.append(f"  - {a} is behind {b}")

        if rel_lines:
            base += "\nSpatial relations:\n" + "\n".join(rel_lines)
        return base

    # ------------------------------------------------------------------
    # JSON export (for web visualizer)
    # ------------------------------------------------------------------

    def to_dict(self, distance_threshold: float = 1.5) -> Dict[str, Any]:
        """Return full state as a JSON-serializable dict."""
        relations = self.compute_relations(distance_threshold)
        rel_list = []
        for rel_type, pairs in relations.items():
            for a, b, meta in pairs:
                rel_list.append({"type": rel_type, "from": a, "to": b, **meta})
        return {
            "explored_steps": self.explored_steps,
            "detected_objects": dict(self.detected_objects),
            "vqa_results": list(self.vqa_results),
            "relations": rel_list,
        }

    def save_json(self, path: str, distance_threshold: float = 1.5) -> None:
        """Write state to a JSON file (for visualizer to read)."""
        import json
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(distance_threshold), f, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_object(self, name: str) -> Dict[str, Any]:
        key = name.strip().lower()
        if key not in self.detected_objects:
            self.detected_objects[key] = {
                "position": None,
                "size": None,
                "step": None,
                "image_path": "",
                "crop_paths": [],
                "bbox_2d": None,
            }
        return self.detected_objects[key]

    def _infer_crop_object(self, args: Dict[str, Any]) -> Optional[str]:
        bbox = args.get("bounding_box") or args.get("bbox")
        if bbox and "object" in args:
            return args["object"]
        for name, info in self.detected_objects.items():
            if info.get("bbox_2d"):
                return name
        return None

    def _infer_crop_object_from_code(self, code_text: str) -> Optional[str]:
        obj = self._extract_tool_arg(code_text, "object")
        if obj:
            return obj
        for name in self.detected_objects:
            if name in code_text.lower():
                return name
        return None

    @staticmethod
    def _extract_xyz(raw: Any) -> Optional[List[float]]:
        if isinstance(raw, (list, tuple)):
            flat = raw
            while isinstance(flat, (list, tuple)) and len(flat) > 0 and isinstance(flat[0], (list, tuple)):
                flat = flat[0]
            try:
                return [float(x) for x in flat[:3]]
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _extract_tool_arg(code_text: str, arg_name: str) -> Optional[str]:
        pattern = rf"""{arg_name}\s*=\s*['\"]([^'\"]+)['\"]"""
        m = re.search(pattern, code_text)
        if m:
            return m.group(1)
        pattern = rf"""{arg_name}:\s*str\s*=\s*['\"]([^'\"]+)['\"]"""
        m = re.search(pattern, code_text)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _parse_3d_result(obs: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """Parse center and size from ObjectLocation3D observation string.

        Expected format: center = [[x, y, z]], size = [[l, w, h]]
        Or: ([[x, y, z]], [[l, w, h]])
        """
        center, size = None, None

        # Try to find center = [[...]]
        m = re.search(r"center\s*=\s*\[\[([^\]]+)\]\]", obs)
        if m:
            try:
                center = [float(x.strip()) for x in m.group(1).split(",")]
            except ValueError:
                pass

        # Try to find size = [[...]]
        m = re.search(r"size\s*=\s*\[\[([^\]]+)\]\]", obs)
        if m:
            try:
                size = [float(x.strip()) for x in m.group(1).split(",")]
            except ValueError:
                pass

        # Fallback: try tuple format ([[x,y,z]], [[l,w,h]])
        if center is None or size is None:
            nums = re.findall(r"[-+]?\d*\.?\d+", obs)
            if len(nums) >= 6:
                try:
                    center = [float(x) for x in nums[:3]]
                    size = [float(x) for x in nums[3:6]]
                except ValueError:
                    pass

        return center, size

    @staticmethod
    def _parse_2d_result(obs: str) -> Tuple[List[List[int]], List[str]]:
        """Parse bboxes and labels from ObjectLocation2D observation string."""
        bboxes, labels = [], []

        # Try to find bboxes_2d = [[x1, y1, x2, y2]] (nested brackets)
        m = re.search(r"bboxes_2d['\"]?\s*:\s*\[\[([^\]]+)\]\]", obs)
        if m:
            try:
                nums = [int(x.strip()) for x in m.group(1).split(",")]
                if len(nums) == 4:
                    bboxes.append(nums)
            except ValueError:
                pass
        else:
            # Fallback: flat format bboxes_2d = [[x1, y1, x2, y2]]
            m = re.search(r"bboxes_2d['\"]?\s*:\s*\[([^\]]*)\]", obs)
            if m:
                for bbox_match in re.finditer(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", m.group(1)):
                    bboxes.append([int(bbox_match.group(i)) for i in range(1, 5)])

        # Try to find labels = [...]
        m = re.search(r"labels['\"]?\s*:\s*\[([^\]]*)\]", obs)
        if m:
            for label_match in re.finditer(r"['\"]([^'\"]+)['\"]", m.group(1)):
                labels.append(label_match.group(1))

        return bboxes, labels

    @staticmethod
    def _parse_crop_paths(obs: str) -> List[str]:
        """Parse file paths from ObjectCrop observation string."""
        paths = re.findall(r"['\"]([^'\"]*\.(?:jpg|png|jpeg))['\"]", obs, re.IGNORECASE)
        return paths
