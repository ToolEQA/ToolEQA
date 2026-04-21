"""
Helper entrypoint for preparing a ToolEQA->verl run.

For Qwen2.5-VL style controller models, prefer the GRPO config because vanilla
PPO requires a critic/value head that the upstream multimodal checkpoints do not
provide by default.
"""

import argparse
from pathlib import Path

from .dataset_builder import build_verl_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/ToolTrajectory/trainval.json")
    parser.add_argument("--output", default="src/train/RFT/verl_adapter/data/tooleqa_train.jsonl")
    parser.add_argument("--config", default="src/train/RFT/verl_adapter/configs/grpo_tooleqa.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    build_verl_dataset(args.source, args.output, limit=args.limit)
    config_path = str(Path(args.config).parent)
    config_name = Path(args.config).stem
    print("Dataset preparation complete.")
    print("Launch template:")
    print(
        "python -m verl.trainer.main_ppo "
        f"--config-path={config_path} "
        f"--config-name={config_name}"
    )


if __name__ == "__main__":
    main()
