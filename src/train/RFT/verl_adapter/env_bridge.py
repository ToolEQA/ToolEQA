import copy
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional


class ToolEQAEnvBridge:
    """
    Thin runtime bridge between ToolEQA tools/environment and the verl agent loop.

    This class deliberately keeps planner and executor fixed. The RL policy only
    decides the next controller action.
    """

    def __init__(self, cfg_path: str, gpu_id: int = 0, open_vocab: bool = True, debug: bool = False):
        self.cfg_path = cfg_path
        self.gpu_id = gpu_id
        self.open_vocab = open_vocab
        self.debug = debug
        self.args = Namespace(cfg=cfg_path, open_vocab=open_vocab)
        self.toolbox = None
        self.tools: Dict[str, Any] = {}
        self.sample: Optional[Dict[str, Any]] = None
        self.trace: List[Dict[str, Any]] = []
        self.evidence_memory: Dict[str, Any] = {}
        self.done = False
        self.final_answer = None

    def _lazy_init_toolbox(self) -> None:
        if self.toolbox is not None:
            return
        if self.debug:
            self.toolbox = self._build_mock_toolbox()
        else:
            # Only set CUDA_VISIBLE_DEVICES if not already configured by veRL
            existing = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if existing.strip() in {"", "-1"}:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            elif str(self.gpu_id) not in existing.split(","):
                os.environ["CUDA_VISIBLE_DEVICES"] = existing + "," + str(self.gpu_id)
            from src.tools.tool_box import get_tool_box

            self.toolbox = get_tool_box(debug=self.debug, gpu_id=self.gpu_id, args=self.args)
        self.tools = {tool.name: tool for tool in self.toolbox}

    def _build_mock_toolbox(self):
        class MockGoNextPointTool:
            name = "GoNextPointTool"

            def __init__(self):
                self.cur_rgb_path = "./cache/init_rgb.png"
                self.step_idx = 0

            def initialize(self, data):
                self.cur_rgb_path = "./cache/init_rgb.png"
                self.step_idx = 0

            def __call__(self, direction):
                self.step_idx += 1
                self.cur_rgb_path = f"./cache/mock_step_{self.step_idx}.png"
                return self.cur_rgb_path

        class MockObjectLocation2D:
            name = "ObjectLocation2D"

            def __call__(self, object, image_path):
                return {"bboxes_2d": [[10, 20, 110, 180]], "labels": [object], "image_path": image_path}

        class MockObjectLocation3D:
            name = "ObjectLocation3D"

            def __call__(self, object, image_path):
                return ([[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]])

        class MockObjectCrop:
            name = "ObjectCrop"

            def __call__(self, bounding_box, image_path):
                return ["./cache/mock_crop_0.png"]

        class MockVisualQATool:
            name = "VisualQATool"

            def __call__(self, question, image_path="", image_paths=""):
                return f"Mock VQA answer for: {question}"

        class MockFinalAnswerTool:
            name = "final_answer"

            def __call__(self, answer=None, **kwargs):
                return answer if answer is not None else kwargs.get("final_answer")

        return [
            MockVisualQATool(),
            MockObjectLocation2D(),
            MockObjectLocation3D(),
            MockGoNextPointTool(),
            MockFinalAnswerTool(),
            MockObjectCrop(),
        ]

    def reset(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        self._lazy_init_toolbox()
        self.sample = copy.deepcopy(sample)
        self.trace = []
        self.done = False
        self.final_answer = None
        self.evidence_memory = {
            "found_objects": set(),
            "tool_calls": [],
            "answered": False,
            "last_vqa_answer": None,
            "last_bbox": None,
            "last_crop_paths": [],
        }
        if not self.debug:
            scene_name = sample.get("scene")
            if scene_name:
                self._load_scene(scene_name)
        for tool in self.toolbox:
            if hasattr(tool, "initialize"):
                tool.initialize(self.sample)
        return self.get_state()

    def get_current_image_path(self) -> Optional[str]:
        tool = self.tools.get("GoNextPointTool")
        return getattr(tool, "cur_rgb_path", None) if tool else None

    def get_state(self) -> Dict[str, Any]:
        return {
            "question": self.sample["question"] if self.sample else None,
            "plan": self.sample.get("plan", "") if self.sample else "",
            "image_path": self.get_current_image_path(),
            "evidence_memory": self._serialize_memory(),
            "done": self.done,
            "final_answer": self.final_answer,
        }

    def _serialize_memory(self) -> Dict[str, Any]:
        memory = dict(self.evidence_memory)
        memory["found_objects"] = sorted(memory["found_objects"])
        return memory

    def _update_memory(self, action_type: str, args: Dict[str, Any], result: Any) -> None:
        self.evidence_memory["tool_calls"].append({"action_type": action_type, "args": args})
        if action_type in {"ObjectLocation2D", "ObjectLocation3D"}:
            obj_name = args.get("object")
            if obj_name:
                self.evidence_memory["found_objects"].add(obj_name)
            self.evidence_memory["last_bbox"] = result
        elif action_type == "ObjectCrop":
            self.evidence_memory["last_crop_paths"] = result if isinstance(result, list) else [result]
        elif action_type == "VisualQA":
            self.evidence_memory["last_vqa_answer"] = result
        elif action_type == "FinalAnswer":
            self.evidence_memory["answered"] = True

    def _execute_named_tool(self, tool_name: str, kwargs: Dict[str, Any]) -> Any:
        tool = self.tools[tool_name]
        return tool(**kwargs)

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self.sample is None:
            raise RuntimeError("Bridge must be reset before execute_action.")
        if self.done:
            return {
                "observation": "Episode already finished.",
                "done": True,
                "state": self.get_state(),
            }

        action_type = action["action_type"]
        args = dict(action.get("args", {}))
        image_path = self.get_current_image_path()

        if action_type == "Navigate":
            result = self._execute_named_tool(
                "GoNextPointTool",
                {"direction": args.get("direction", args.get("value"))},
            )
            observation = f"Moved to next viewpoint. Current image saved at {result}."
        elif action_type == "ObjectLocation2D":
            args.setdefault("image_path", image_path)
            result = self._execute_named_tool("ObjectLocation2D", args)
            observation = f"2D localization result: {result}"
        elif action_type == "ObjectLocation3D":
            args.setdefault("image_path", image_path)
            result = self._execute_named_tool("ObjectLocation3D", args)
            observation = f"3D localization result: {result}"
        elif action_type == "ObjectCrop":
            args.setdefault("image_path", image_path)
            if "bounding_box" not in args and "bbox" in args:
                args["bounding_box"] = args["bbox"]
            result = self._execute_named_tool("ObjectCrop", args)
            observation = f"Object crop saved to {result}"
        elif action_type == "VisualQA":
            if "image_paths" not in args:
                crop_paths = self.evidence_memory.get("last_crop_paths") or []
                args["image_paths"] = crop_paths if crop_paths else [image_path]
            result = self._execute_named_tool("VisualQATool", args)
            observation = str(result)
        elif action_type == "FinalAnswer":
            answer = args.get("answer", args.get("value"))
            result = self._execute_named_tool("final_answer", {"answer": answer})
            self.final_answer = result
            self.done = True
            observation = str(result)
        else:
            raise ValueError(f"Unsupported action_type: {action_type}")

        self._update_memory(action_type, args, result)
        step_record = {
            "action_type": action_type,
            "args": args,
            "result": result,
            "observation": observation,
            "image_path": self.get_current_image_path(),
        }
        self.trace.append(step_record)
        return {
            "observation": observation,
            "done": self.done,
            "state": self.get_state(),
            "step_record": step_record,
        }

    def get_episode_trace(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.trace)

    def close(self) -> None:
        """Release Habitat resources between episodes."""
        if self.toolbox is not None and not self.debug:
            for tool in self.toolbox:
                if hasattr(tool, "close"):
                    try:
                        tool.close()
                    except Exception:
                        pass
                elif hasattr(tool, "sim") and hasattr(tool.sim, "close"):
                    try:
                        tool.sim.close()
                    except Exception:
                        pass
            self.toolbox = None
            self.tools = {}
        self.trace = []
        self.evidence_memory = {}
        self.done = False
        self.final_answer = None

    def _load_scene(self, scene_name: str) -> None:
        """Load Habitat scene for a new episode. Only relevant in non-debug mode."""
        if self.debug:
            return
        go_tool = self.tools.get("GoNextPointTool")
        if go_tool is None:
            return
        if hasattr(go_tool, "load_scene"):
            go_tool.load_scene(scene_name)
        elif hasattr(go_tool, "sim") and hasattr(go_tool.sim, "reconfigure"):
            go_tool.sim.reconfigure()
