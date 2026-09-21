"""Resolve the RFT Hydra config without importing VERL, vLLM, or CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        default=str(root / "src/train/RFT/verl_adapter/configs"),
    )
    parser.add_argument("--config-name", default="evidence_grpo")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    with initialize_config_dir(config_dir=str(Path(args.config_dir).resolve()), version_base=None):
        config = compose(config_name=args.config_name, overrides=args.overrides)
    print(OmegaConf.to_yaml(config, resolve=True))


if __name__ == "__main__":
    main()
