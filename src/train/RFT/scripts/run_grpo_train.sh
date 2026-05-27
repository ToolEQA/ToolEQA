#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
RFT_DIR="$ROOT_DIR/src/train/RFT"
ADAPTER_DIR="$RFT_DIR/verl_adapter"
CONFIG_PATH="$ADAPTER_DIR/configs"

PYTHON_BIN="${PYTHON_BIN:-/home/zml/programs/miniconda3/envs/verl-tooleqa/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mynvme0/models/Qwen/Qwen2.5-VL-7B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$ADAPTER_DIR/data/tooleqa_train.jsonl}"
VAL_FILE="${VAL_FILE:-$TRAIN_FILE}"
CONFIG_NAME="${CONFIG_NAME:-grpo_tooleqa}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
NNODES="${NNODES:-1}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-4}"
ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-controller-grpo}"
PROJECT_NAME="${PROJECT_NAME:-tooleqa-verl}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
TEST_FREQ="${TEST_FREQ:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
VAL_ONLY="${VAL_ONLY:-false}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.85}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-512}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-2}"
TORCH_CUDA_ARCH_LIST_VALUE="${TORCH_CUDA_ARCH_LIST:-8.9}"
DRY_RUN="${DRY_RUN:-false}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH/$CONFIG_NAME.yaml" ]]; then
    echo "Config file not found: $CONFIG_PATH/$CONFIG_NAME.yaml" >&2
    exit 1
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "Training file not found: $TRAIN_FILE" >&2
    exit 1
fi

if [[ ! -f "$VAL_FILE" ]]; then
    echo "Validation file not found: $VAL_FILE" >&2
    exit 1
fi

PATCHED_MODEL_PATH="$MODEL_PATH"
MODEL_CONFIG_PATH="$MODEL_PATH/config.json"
if [[ -f "$MODEL_CONFIG_PATH" ]]; then
    PATCHED_MODEL_OUTPUT="$("$PYTHON_BIN" - <<'PY' "$MODEL_PATH"
import json
import os
import shutil
import sys

model_path = sys.argv[1]
config_path = os.path.join(model_path, "config.json")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

rope_scaling = config.get("rope_scaling")
needs_patch = isinstance(rope_scaling, dict) and rope_scaling.get("type") == "mrope" and "type" not in rope_scaling

if not needs_patch:
    print(model_path)
    raise SystemExit(0)

patched_dir = os.path.join("/tmp", f"{os.path.basename(model_path)}_vllm_patched")
if not os.path.exists(patched_dir):
    shutil.copytree(model_path, patched_dir, symlinks=True)

patched_config_path = os.path.join(patched_dir, "config.json")
with open(patched_config_path, "r", encoding="utf-8") as f:
    patched = json.load(f)

patched_rope_scaling = dict(patched.get("rope_scaling") or {})
if "type" in patched_rope_scaling:
    patched_rope_scaling["type"] = patched_rope_scaling.pop("type")
elif "rope_type" not in patched_rope_scaling:
    patched_rope_scaling["rope_type"] = "mrope"
patched["rope_scaling"] = patched_rope_scaling

with open(patched_config_path, "w", encoding="utf-8") as f:
    json.dump(patched, f, ensure_ascii=False, indent=2)
    f.write("\n")

print(patched_dir)
PY
)"
    PATCHED_MODEL_PATH="$PATCHED_MODEL_OUTPUT"
fi

IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "$CUDA_VISIBLE_DEVICES_VALUE"
VISIBLE_GPU_COUNT="${#CUDA_DEVICE_ARRAY[@]}"
REQUIRED_GPU_COUNT=$((TRAIN_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE))
if (( REQUIRED_GPU_COUNT == 0 )); then
    REQUIRED_GPU_COUNT=1
fi
# For colocated/overlapping configs, use max instead of sum
if (( VISIBLE_GPU_COUNT < REQUIRED_GPU_COUNT )); then
    MAX_GPU=$((TRAIN_GPUS_PER_NODE > ROLLOUT_GPUS_PER_NODE ? TRAIN_GPUS_PER_NODE : ROLLOUT_GPUS_PER_NODE))
    if (( MAX_GPU == 0 )); then MAX_GPU=1; fi
    if (( VISIBLE_GPU_COUNT >= MAX_GPU )); then
        echo "Warning: TRAIN_GPUS_PER_NODE($TRAIN_GPUS_PER_NODE) + ROLLOUT_GPUS_PER_NODE($ROLLOUT_GPUS_PER_NODE) > visible GPUs($VISIBLE_GPU_COUNT)" >&2
        echo "  Training and rollout will share GPUs." >&2
        REQUIRED_GPU_COUNT="$MAX_GPU"
    else
        echo "Not enough visible GPUs." >&2
        echo "  visible: $VISIBLE_GPU_COUNT ($CUDA_VISIBLE_DEVICES_VALUE)" >&2
        echo "  required: $MAX_GPU = max(TRAIN_GPUS_PER_NODE($TRAIN_GPUS_PER_NODE), ROLLOUT_GPUS_PER_NODE($ROLLOUT_GPUS_PER_NODE))" >&2
        exit 1
    fi
fi

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST_VALUE"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export HABITAT_SIM_LOG="${HABITAT_SIM_LOG:-quiet}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"

CMD=(
    "$PYTHON_BIN" -m verl.experimental.one_step_off_policy.main_ppo
    --config-path="$CONFIG_PATH"
    --config-name="$CONFIG_NAME"
    actor_rollout_ref.model.path="$PATCHED_MODEL_PATH"
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL"
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_BATCHED_TOKENS"
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS"
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXPERIMENT_NAME"
    trainer.total_epochs="$TOTAL_EPOCHS"
    trainer.save_freq="$SAVE_FREQ"
    trainer.test_freq="$TEST_FREQ"
    trainer.val_only="$VAL_ONLY"
    trainer.n_gpus_per_node="$TRAIN_GPUS_PER_NODE"
    trainer.nnodes="$NNODES"
    rollout.n_gpus_per_node="$ROLLOUT_GPUS_PER_NODE"
    rollout.nnodes="$NNODES"
    actor_rollout_ref.hybrid_engine=false
    algorithm.adv_estimator=grpo
    data.train_files="[$TRAIN_FILE]"
    data.val_files="[$VAL_FILE]"
)

if [[ "$#" -gt 0 ]]; then
    CMD+=("$@")
fi

echo "Launching ToolEQA GRPO training"
echo "  root: $ROOT_DIR"
echo "  config: $CONFIG_PATH/$CONFIG_NAME.yaml"
echo "  model: $MODEL_PATH"
echo "  patched_model: $PATCHED_MODEL_PATH"
echo "  train: $TRAIN_FILE"
echo "  val: $VAL_FILE"
echo "  gpus: $CUDA_VISIBLE_DEVICES"
echo "  train_gpus_per_node: $TRAIN_GPUS_PER_NODE"
echo "  rollout_gpus_per_node: $ROLLOUT_GPUS_PER_NODE"
echo "  python: $PYTHON_BIN"
echo "  adv_estimator: grpo"

if [[ "$DRY_RUN" == "true" ]]; then
    printf 'DRY_RUN command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${CMD[@]}"
