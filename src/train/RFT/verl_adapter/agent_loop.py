from typing import Any, Callable, Dict, List, Optional

from .action_parser import parse_action_output
from .env_bridge import ToolEQAEnvBridge
from .reward_manager import ToolEQARewardManager
from .tool_wrappers import set_tool_env_bridge


DEFAULT_SYSTEM_PROMPT = """You are the ToolEQA controller.
At each turn, decide the next action that best improves evidence collection.
Prefer JSON output with this schema:
{"action_type": "...", "args": {...}}
Available action_type:
- Navigate
- ObjectLocation2D
- ObjectLocation3D
- ObjectCrop
- VisualQA
- FinalAnswer
Only output one action per turn."""


class ToolEQAAgentLoop:
    """
    A user-defined multi-turn loop compatible with verl's agent-loop idea.

    The loop can run with any callable model_generate(messages, image_paths) -> str.
    """

    def __init__(
        self,
        bridge: ToolEQAEnvBridge,
        reward_manager: Optional[ToolEQARewardManager] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 20,
    ):
        self.bridge = bridge
        self.reward_manager = reward_manager or ToolEQARewardManager()
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        set_tool_env_bridge(bridge)

    def _build_initial_messages(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {sample['question']}\n\n"
                    f"Planner Plan:\n{sample.get('plan', '')}\n\n"
                    f"Current image path: {self.bridge.get_current_image_path()}\n"
                    "Choose the next action."
                ),
            },
        ]

    def run(self, sample: Dict[str, Any], model_generate: Callable[..., str]) -> Dict[str, Any]:
        self.bridge.reset(sample)
        messages = self._build_initial_messages(sample)

        for _ in range(self.max_turns):
            image_path = self.bridge.get_current_image_path()
            raw_output = model_generate(messages=messages, image_paths=[image_path] if image_path else [])
            action = parse_action_output(raw_output)
            step_result = self.bridge.execute_action(action)

            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "tool", "content": step_result["observation"]})

            if step_result["done"]:
                break

        trace = self.bridge.get_episode_trace()
        reward = self.reward_manager(sample, trace, self.bridge.final_answer)
        return {
            "messages": messages,
            "trace": trace,
            "final_answer": self.bridge.final_answer,
            "reward": reward,
        }
