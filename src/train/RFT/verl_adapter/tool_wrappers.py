from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .env_bridge import ToolEQAEnvBridge


def _default_schema(name: str, description: str, properties: Dict[str, Any], required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class ToolEQABaseWrapper(BaseTool):
    action_type: str = ""
    schema_dict: Dict[str, Any] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        config = config or {}
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate(self.schema_dict)
        super().__init__(config=config, tool_schema=tool_schema)

    def _get_bridge(self, agent_data) -> ToolEQAEnvBridge:
        bridge = agent_data.extra_fields.get("tooleqa_bridge")
        if bridge is not None:
            return bridge

        cfg_path = self.config.get("cfg_path", "config/react-eqa.yaml")
        gpu_id = int(self.config.get("gpu_id", 0))
        open_vocab = bool(self.config.get("open_vocab", True))
        debug = bool(self.config.get("debug", False))
        bridge = ToolEQAEnvBridge(cfg_path=cfg_path, gpu_id=gpu_id, open_vocab=open_vocab, debug=debug)

        sample_info = copy.deepcopy(agent_data.extra_fields.get("sample_info", {}))
        bridge.reset(sample_info)
        agent_data.extra_fields["tooleqa_bridge"] = bridge
        return bridge

    async def create(self, instance_id: Optional[str] = None, **kwargs):
        return await super().create(instance_id=instance_id, **kwargs)

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs):
        agent_data = kwargs["agent_data"]
        bridge = self._get_bridge(agent_data)
        result = bridge.execute_action({"action_type": self.action_type, "args": parameters})
        text = result["observation"]
        agent_data.extra_fields["tooleqa_trace"] = bridge.get_episode_trace()
        if result["done"]:
            agent_data.extra_fields["tooleqa_final_answer"] = bridge.final_answer
        return ToolResponse(text=text), 0.0, {"done": result["done"]}

    async def release(self, instance_id: str, **kwargs) -> None:
        return None


class NavigateTool(ToolEQABaseWrapper):
    action_type = "Navigate"
    schema_dict = _default_schema(
        "Navigate",
        "Navigate to the next viewpoint in the ToolEQA environment.",
        {"direction": {"type": "string", "enum": ["move_forward", "turn_left", "turn_right", "turn_around"]}},
        ["direction"],
    )


class ObjectLocation2DTool(ToolEQABaseWrapper):
    action_type = "ObjectLocation2D"
    schema_dict = _default_schema(
        "ObjectLocation2D",
        "Locate an object in the current image and return 2D boxes.",
        {"object": {"type": "string"}},
        ["object"],
    )


class ObjectLocation3DTool(ToolEQABaseWrapper):
    action_type = "ObjectLocation3D"
    schema_dict = _default_schema(
        "ObjectLocation3D",
        "Estimate the 3D location and size of an object.",
        {"object": {"type": "string"}},
        ["object"],
    )


class ObjectCropTool(ToolEQABaseWrapper):
    action_type = "ObjectCrop"
    schema_dict = _default_schema(
        "ObjectCrop",
        "Crop an object region from the current image.",
        {"bounding_box": {"type": "array", "items": {"type": "number"}}},
        ["bounding_box"],
    )


class VisualQAToolWrapper(ToolEQABaseWrapper):
    action_type = "VisualQA"
    schema_dict = _default_schema(
        "VisualQA",
        "Ask a visual question about the current or cropped image.",
        {"question": {"type": "string"}},
        ["question"],
    )


class FinalAnswerToolWrapper(ToolEQABaseWrapper):
    action_type = "FinalAnswer"
    schema_dict = _default_schema(
        "FinalAnswer",
        "End the episode with the final answer.",
        {"answer": {"type": "string"}},
        ["answer"],
    )
