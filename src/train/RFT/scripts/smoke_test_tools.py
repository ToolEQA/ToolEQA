#!/usr/bin/env python3
"""Run one real EQA-RT/HM3D call through every rollout tool."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.tools.tool_box import get_tool_box


def jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, list, tuple, dict)):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_sample(path: Path, index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                row = json.loads(line)
                return row.get("extra_info", row)
    raise IndexError(f"Sample index {index} is outside {path}")


def run_tool(
    results: dict[str, Any],
    name: str,
    function: Callable[[], Any],
    validator: Callable[[Any], None],
) -> Any:
    started = time.perf_counter()
    print(f"[RUN] {name}", flush=True)
    try:
        output = function()
        validator(output)
        results[name] = {
            "status": "pass",
            "latency_s": round(time.perf_counter() - started, 3),
            "output": jsonable(output),
        }
        print(f"[PASS] {name} ({results[name]['latency_s']}s)", flush=True)
        return output
    except Exception as error:
        results[name] = {
            "status": "fail",
            "latency_s": round(time.perf_counter() - started, 3),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        print(f"[FAIL] {name}: {results[name]['error']}", flush=True)
        return None


def require_image(path: Any) -> None:
    if not isinstance(path, str) or not os.path.isfile(path):
        raise AssertionError(f"Expected an existing image path, got {path!r}")
    with Image.open(path) as image:
        image.verify()


def require_2d(result: Any) -> None:
    if not isinstance(result, dict) or "bboxes_2d" not in result:
        raise AssertionError(f"Unexpected 2D result: {result!r}")
    boxes = jsonable(result["bboxes_2d"])
    if not isinstance(boxes, list) or any(not isinstance(box, list) or len(box) != 4 for box in boxes):
        raise AssertionError(f"Malformed 2D boxes: {boxes!r}")


def require_3d(result: Any) -> None:
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise AssertionError(f"3D tool must return exactly (centers, sizes), got {result!r}")
    centers, sizes = jsonable(result)
    if not isinstance(centers, list) or not isinstance(sizes, list) or len(centers) != len(sizes):
        raise AssertionError(f"Malformed 3D result: {result!r}")
    if any(len(center) != 3 for center in centers) or any(len(size) != 3 for size in sizes):
        raise AssertionError(f"Malformed 3D center/size vectors: {result!r}")


def require_crops(result: Any) -> None:
    if not isinstance(result, list) or not result:
        raise AssertionError(f"Expected at least one crop path, got {result!r}")
    for path in result:
        require_image(path)


def require_text(result: Any) -> None:
    if not isinstance(result, str) or not result.strip():
        raise AssertionError(f"Expected non-empty text, got {result!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/react-eqa.yaml"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--channel", type=int, default=0, help="DetAny shared-memory channel")
    parser.add_argument("--agent-gpu", type=int, default=2, help="CUDA index for Habitat/VQA")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sample = load_sample(args.data, args.index)
    target = str((sample.get("related_objects") or [{"name": "chair"}])[0].get("name", "chair"))
    results: dict[str, Any] = {
        "sample_id": sample.get("sample_id"),
        "scene": sample.get("scene"),
        "target": target,
        "data": str(args.data.resolve()),
        "tools": {},
    }
    toolbox = []
    try:
        started = time.perf_counter()
        toolbox = get_tool_box(
            debug=False,
            gpu_id=args.channel,
            args=SimpleNamespace(cfg=str(args.config.resolve()), open_vocab=True),
        )
        tools = {tool.name: tool for tool in toolbox}
        expected = {
            "GoNextPointTool",
            "ObjectLocation2D",
            "ObjectLocation3D",
            "ObjectCrop",
            "VisualQATool",
            "final_answer",
        }
        if set(tools) != expected:
            raise AssertionError(f"Toolbox mismatch: expected {sorted(expected)}, got {sorted(tools)}")
        for tool in toolbox:
            if tool.name in {"GoNextPointTool", "VisualQATool"}:
                tool.gpu_id = args.agent_gpu
            if hasattr(tool, "initialize"):
                tool.initialize(sample)
        results["initialization"] = {
            "status": "pass",
            "latency_s": round(time.perf_counter() - started, 3),
        }

        navigation = tools["GoNextPointTool"]
        initial_image = navigation.cur_rgb_path
        require_image(initial_image)
        image_path = run_tool(
            results["tools"],
            "GoNextPointTool",
            lambda: navigation.forward("turn_left"),
            require_image,
        ) or initial_image

        result_2d = run_tool(
            results["tools"],
            "ObjectLocation2D",
            lambda: tools["ObjectLocation2D"].forward(object=target, image_path=image_path),
            require_2d,
        )

        run_tool(
            results["tools"],
            "ObjectLocation3D",
            lambda: tools["ObjectLocation3D"].forward(object=target, image_path=image_path),
            require_3d,
        )

        with Image.open(image_path) as image:
            width, height = image.size
        boxes = jsonable(result_2d.get("bboxes_2d", [])) if isinstance(result_2d, dict) else []
        if boxes:
            box = [max(0, int(value)) for value in boxes[0]]
            box[2] = min(width, max(box[0] + 1, box[2]))
            box[3] = min(height, max(box[1] + 1, box[3]))
        else:
            box = [0, 0, max(1, min(128, width)), max(1, min(128, height))]
        crop_paths = run_tool(
            results["tools"],
            "ObjectCrop",
            lambda: tools["ObjectCrop"].forward(bounding_box=box, image_path=image_path),
            require_crops,
        )

        vqa_image = crop_paths[0] if crop_paths else image_path
        vqa_output = run_tool(
            results["tools"],
            "VisualQATool",
            lambda: tools["VisualQATool"].forward(
                question="Briefly describe what is visible in this image.",
                image_paths=[vqa_image],
            ),
            require_text,
        )
        if isinstance(vqa_output, str) and len(vqa_output) > 1000:
            results["tools"]["VisualQATool"]["output"] = vqa_output[:1000] + "..."

        run_tool(
            results["tools"],
            "final_answer",
            lambda: tools["final_answer"].forward(answer="A"),
            lambda output: None if output == "A" else (_ for _ in ()).throw(
                AssertionError(f"Expected 'A', got {output!r}")
            ),
        )
    except Exception as error:
        results["initialization"] = {
            "status": "fail",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    finally:
        for tool in toolbox:
            backend = getattr(tool, "eqa_modeling", None)
            simulator = getattr(backend, "simulator", None) if backend is not None else None
            if simulator is not None:
                try:
                    simulator.close()
                except Exception:
                    pass

    tool_results = results["tools"]
    results["summary"] = {
        "passed": sum(item.get("status") == "pass" for item in tool_results.values()),
        "failed": sum(item.get("status") == "fail" for item in tool_results.values()),
        "expected": 6,
    }
    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    print(rendered, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if results.get("initialization", {}).get("status") == "pass" and results["summary"]["passed"] == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
