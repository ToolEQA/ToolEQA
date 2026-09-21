#!/usr/bin/env bash
set -euo pipefail

TOOLEQA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zml/programs/miniconda3/envs/verl-tooleqa/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mynvme0/models/Qwen/Qwen3-VL-8B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-${TOOLEQA_ROOT}/src/train/RFT/data/train_reward_eligible.jsonl}"
VAL_FILE="${VAL_FILE:-${TOOLEQA_ROOT}/src/train/RFT/data/validation_reward_eligible.jsonl}"
RAY_VISIBLE_GPUS="${RAY_VISIBLE_GPUS:-0,1,2,3}"
TRAIN_GPUS="${TRAIN_GPUS:-3}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}"
TOOLEQA_TOOL_GPU_ID="${TOOLEQA_TOOL_GPU_ID:-0}"
TOOLEQA_AGENT_GPU_ID="${TOOLEQA_AGENT_GPU_ID:-3}"
CHECKPOINT_BACKEND="${CHECKPOINT_BACKEND:-nccl}"
RFT_OUTPUT_ROOT="${RFT_OUTPUT_ROOT:-/mynvme0/ToolEQA_RFT}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-evidence-grpo-qwen3-vl-8b-instruct-thought-code}"
for override in "$@"; do
  case "${override}" in
    trainer.experiment_name=*) EXPERIMENT_NAME="${override#trainer.experiment_name=}" ;;
  esac
done
RUN_DIR="${RFT_OUTPUT_ROOT}/${EXPERIMENT_NAME}"
TOOLEQA_TRAJECTORY_DIR="${TOOLEQA_TRAJECTORY_DIR:-${RUN_DIR}/trajectories}"

export TOOLEQA_ROOT MODEL_PATH TOOLEQA_TOOL_GPU_ID TOOLEQA_AGENT_GPU_ID
export RFT_OUTPUT_ROOT EXPERIMENT_NAME TOOLEQA_TRAJECTORY_DIR
export CUDA_VISIBLE_DEVICES="${RAY_VISIBLE_GPUS}"
export TOOLEQA_LIMIT_NVML_TO_VISIBLE="${TOOLEQA_LIMIT_NVML_TO_VISIBLE:-1}"
# CUDA skips the two broken PCI devices at NVML indices 4 and 5, so CUDA
# ordinal 4 is the healthy L40 reported by NVML as index 6.
export TOOLEQA_NVML_INDEX_MAP="${TOOLEQA_NVML_INDEX_MAP:-4:6}"
if [[ -z "${TOOLEQA_DISTRIBUTED_BACKEND:-}" ]]; then
  GPU_LIST_OUTPUT="$(nvidia-smi -L 2>&1 || true)"
  if [[ "${GPU_LIST_OUTPUT}" == *"Unable to determine the device handle"* ]]; then
    export TOOLEQA_DISTRIBUTED_BACKEND=gloo
    export VERL_FSDP_WEIGHT_SYNC_CPU=1
    export VERL_WEIGHT_TRANSFER_SHM=1
    export TOOLEQA_RANK_DEVICE_MAP=1
    # Broken NVML makes Ray assign accelerator id 0 to every fractional
    # training actor. Keep the rank-specific masks injected by verl instead.
    export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
    CHECKPOINT_BACKEND=gloo
    echo "WARNING: broken NVML indices detected; using slow Gloo CUDA collectives instead of NCCL"
  fi
fi
export PYTHONPATH="${TOOLEQA_ROOT}/src/train/RFT/runtime:${TOOLEQA_ROOT}:${TOOLEQA_ROOT}/third_party/verl${PYTHONPATH:+:${PYTHONPATH}}"

PREFLIGHT_ARGS=(--model "${MODEL_PATH}" --train "${TRAIN_FILE}" --validation "${VAL_FILE}" --channel "${TOOLEQA_TOOL_GPU_ID}")
if [[ "${OFFICIAL_TEST:-0}" == "1" ]]; then
  if [[ " $* " != *" trainer.val_only=true "* ]]; then
    echo "OFFICIAL_TEST requires trainer.val_only=true" >&2
    exit 2
  fi
  PREFLIGHT_ARGS+=(--official-test)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--skip-scenes)
elif [[ "${SKIP_DETANY_CHECK:-0}" != "1" ]]; then
  PREFLIGHT_ARGS+=(--require-detany)
fi
"${PYTHON_BIN}" -m src.train.RFT.preflight "${PREFLIGHT_ARGS[@]}"

OVERRIDES=(
  "data.train_files=[${TRAIN_FILE}]"
  "data.val_files=[${VAL_FILE}]"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.rollout.checkpoint_engine.backend=${CHECKPOINT_BACKEND}"
  "actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=src.train.RFT.verl_adapter.gloo_checkpoint_engine"
  "trainer.n_gpus_per_node=${TRAIN_GPUS}"
  "rollout.n_gpus_per_node=${ROLLOUT_GPUS}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.rollout_data_dir=${RUN_DIR}/rollouts"
  "trainer.validation_data_dir=${RUN_DIR}/validation"
  "trainer.default_local_dir=${RUN_DIR}/checkpoints"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exec "${PYTHON_BIN}" -m src.train.RFT.resolve_config "${OVERRIDES[@]}" "$@"
fi
exec "${PYTHON_BIN}" -m verl.experimental.one_step_off_policy.main_ppo \
  --config-path "${TOOLEQA_ROOT}/src/train/RFT/verl_adapter/configs" \
  --config-name evidence_grpo \
  "${OVERRIDES[@]}" "$@"
