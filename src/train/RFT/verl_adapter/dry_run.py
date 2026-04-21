import argparse
import json
from typing import Dict, List

from .agent_loop import ToolEQAAgentLoop
from .env_bridge import ToolEQAEnvBridge


class ScriptedModel:
    def __init__(self, outputs: List[str]):
        self.outputs = outputs
        self.index = 0

    def __call__(self, messages, image_paths):
        if self.index >= len(self.outputs):
            return '{"action_type":"FinalAnswer","args":{"answer":"C"}}'
        output = self.outputs[self.index]
        self.index += 1
        return output


def load_sample(source: str, sample_index: int) -> Dict:
    data = json.load(open(source, "r", encoding="utf-8"))
    return data[sample_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/ToolTrajectory/trainval.json")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--cfg", default="config/react-eqa.yaml")
    args = parser.parse_args()

    sample = load_sample(args.source, args.sample_index)
    bridge = ToolEQAEnvBridge(cfg_path=args.cfg, debug=True)
    loop = ToolEQAAgentLoop(bridge=bridge, max_turns=4)

    scripted_outputs = [
        '{"action_type":"Navigate","args":{"direction":"move_forward"}}',
        '{"action_type":"ObjectLocation2D","args":{"object":"bed"}}',
        '{"action_type":"VisualQA","args":{"question":"Is it occupied?"}}',
        '{"action_type":"FinalAnswer","args":{"answer":"C"}}',
    ]
    result = loop.run(sample, model_generate=ScriptedModel(scripted_outputs))

    print("Dry-run finished.")
    print("Final answer:", result["final_answer"])
    print("Reward:", result["reward"])
    print("Trace length:", len(result["trace"]))
    for idx, step in enumerate(result["trace"]):
        print(f"[{idx}] {step['action_type']} -> {step['observation']}")


if __name__ == "__main__":
    main()
