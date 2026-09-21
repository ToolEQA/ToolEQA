"""Python entry point for audited full Seen/Unseen evaluation.

Preview: python -m src.evaluation.run_open_eval --run-root /path/to/run
Execute explicitly with --execute, after training has finished. Low-level
VERL launch/merge helpers remain in the training scripts directory.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def evaluation_env(run_root: Path):
    env = dict(os.environ)
    env.update(
        PYTHON_BIN=sys.executable, RFT_OUTPUT_ROOT=str(run_root),
        TRAIN_FILE=str(run_root / "data-v2/train.jsonl"),
        RAY_VISIBLE_GPUS="0,1,2,3", TRAIN_GPUS="3", ROLLOUT_GPUS="1",
        TOOLEQA_TOOL_GPU_ID="0", TOOLEQA_AGENT_GPU_ID="3",
        TOOLEQA_DISTRIBUTED_BACKEND="nccl", CHECKPOINT_BACKEND="gloo",
        VERL_FSDP_WEIGHT_SYNC_CPU="1", VERL_WEIGHT_TRANSFER_SHM="1",
        TOOLEQA_LIMIT_NVML_TO_VISIBLE="1", TOOLEQA_NVML_INDEX_MAP="4:6",
        TOOLEQA_RANK_DEVICE_MAP="1", RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1",
        TOOLEQA_SCENE_ROOT=env.get("TOOLEQA_SCENE_ROOT", "/data/zml/datasets/EmbodiedQA/HM3D"),
        TOOLEQA_FROZEN_SERVICE=env.get("TOOLEQA_FROZEN_SERVICE", "http://127.0.0.1:18941"),
        TOOLEQA_FAST_PATCH_EMBED="1", TOOLEQA_NVML_HEALTHY_PREFIX="4",
        PYTHONUNBUFFERED="1", OFFICIAL_TEST="1",
    )
    env.pop("TOOLEQA_RESUME_REWIND_PREFETCH", None)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.pop("PYTORCH_ALLOC_CONF", None)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    guard = Path("/mynvme1/ToolEQA_ICLR2027/speed-tests-20260916/nvml_healthy_prefix.so")
    if not guard.is_file():
        raise FileNotFoundError(f"Required healthy-GPU NVML guard missing: {guard}")
    env["LD_PRELOAD"] = str(guard)
    return env


def execute(run_root: Path):
    from src.evaluation.select_open_checkpoint import select
    from src.evaluation.summarize_rollouts import summarize

    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    # Never contend with this run's active training or another evaluation.
    with (run_root / "pipeline.lock").open("a") as training_lock, \
            (run_root / "evaluation.lock").open("a") as evaluation_lock:
        for lock in (training_lock, evaluation_lock):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Training/evaluation is still active; no evaluation launched") from exc
        for split in ("seen", "unseen"):
            if not (run_root / f"data-v2/{split}.jsonl").is_file():
                raise FileNotFoundError(f"Missing {split} manifest")
            for target in (run_root / f"test-{split}", run_root / f"test-{split}.log",
                           run_root / f"test-{split}-summary.json"):
                if target.exists():
                    raise FileExistsError(f"Refusing to overwrite evaluation: {target}")
        selection = select(run_root)  # Existing strict audit; fail closed on duplicates.
        checkpoint = Path(selection["checkpoint"])
        merged = checkpoint / "actor_huggingface"
        env = evaluation_env(run_root)
        if merged.exists():
            raise FileExistsError(f"Merge target already exists; inspect it before reuse: {merged}")
        with (run_root / "merge-selected.log").open("x") as log:
            subprocess.run(["bash", "src/train/RFT/scripts/merge_fsdp_checkpoint.sh",
                            str(checkpoint), str(merged)], cwd=REPO_ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        with (run_root / "checkpoint-selection.json").open("x") as handle:
            json.dump(selection, handle, indent=2)
        env.update(CHECKPOINT_PATH=str(checkpoint), MERGED_MODEL_PATH=str(merged))
        for split in ("seen", "unseen"):
            env.update(VAL_FILE=str(run_root / f"data-v2/{split}.jsonl"),
                       EXPERIMENT_NAME=f"test-{split}")
            print(f"START full open {split}", flush=True)
            with (run_root / f"test-{split}.log").open("x") as log:
                subprocess.run([
                    "bash", "src/train/RFT/scripts/run_fixed_eval.sh", "checkpoint",
                    f"reward.custom_reward_function.path={REPO_ROOT}/src/evaluation/official_eval.py",
                ], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = summarize(run_root / f"test-{split}/validation/0.jsonl")
            with (run_root / f"test-{split}-summary.json").open("x") as handle:
                json.dump([result], handle, indent=2)
        print("Full evaluation finished; audit sample identities before reporting results.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="Actually launch evaluation; default is preview only")
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not args.execute:
        print(json.dumps({"run_root": str(root), "execute": False,
                          "stages": ["lock and audit", "select joint checkpoint", "merge", "seen", "unseen"],
                          "scorer": "src.evaluation.official_eval.compute_score"}, indent=2))
        return
    execute(root)


if __name__ == "__main__":
    main()
