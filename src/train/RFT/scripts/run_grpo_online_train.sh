#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CONFIG_PATH="$ROOT_DIR/src/train/RFT/verl_adapter/configs"

PYTHON_BIN="${PYTHON_BIN:-/home/zml/programs/miniconda3/envs/verl-tooleqa/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-grpo_tooleqa_online}"
MODEL_PATH="${MODEL_PATH:-/mynvme0/models/Qwen/Qwen2.5-VL-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT_DIR/src/train/RFT/verl_adapter/data/tooleqa_train.jsonl}"
VAL_FILE="${VAL_FILE:-$TRAIN_FILE}"
PROJECT_NAME="${PROJECT_NAME:-tooleqa-verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-controller-grpo-online}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_ONLY="${VAL_ONLY:-false}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.39}"
TOOLEQA_TOOL_GPU_ID="${TOOLEQA_TOOL_GPU_ID:-0}"
TOOLEQA_AGENT_LOOP_NUM_GPUS="${TOOLEQA_AGENT_LOOP_NUM_GPUS:-1}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
export HABITAT_SIM_LOG="${HABITAT_SIM_LOG:-quiet}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export TOOLEQA_TOOL_GPU_ID="$TOOLEQA_TOOL_GPU_ID"
export TOOLEQA_AGENT_LOOP_NUM_GPUS="$TOOLEQA_AGENT_LOOP_NUM_GPUS"

exec "$PYTHON_BIN" -m verl.trainer.main_ppo_sync \
  --config-path="$CONFIG_PATH" \
  --config-name="$CONFIG_NAME" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_only="$VAL_ONLY" \
  trainer.n_gpus_per_node="$GPUS_PER_NODE" \
  trainer.nnodes="$NNODES" \
  data.train_files="[$TRAIN_FILE]" \
  data.val_files="[$VAL_FILE]" \
  "$@"
