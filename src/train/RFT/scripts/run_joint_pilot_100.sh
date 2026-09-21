#!/usr/bin/env bash
set -euo pipefail

TOOLEQA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DEFAULT_STAGE1_CHECKPOINT="/mynvme0/ToolEQA_RFT/eqa-rt-rft-instruct-thought-code-stage1-pilot150/checkpoints/global_step_150"
export MODEL_PATH="${MODEL_PATH:-${DEFAULT_STAGE1_CHECKPOINT}/actor_huggingface}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-eqa-rt-rft-instruct-thought-code-joint-pilot100-n12}"
export TRAIN_FILE="${TRAIN_FILE:-${TOOLEQA_ROOT}/src/train/RFT/data/train_balanced_reward_eligible_450.jsonl}"
export VAL_FILE="${VAL_FILE:-${TOOLEQA_ROOT}/src/train/RFT/data/validation_stratified_reward_eligible_45.jsonl}"
ROLLOUT_N="${ROLLOUT_N:-12}"

if (( ROLLOUT_N % 3 != 0 )); then
  echo "ROLLOUT_N must be divisible by the three FSDP training ranks: ${ROLLOUT_N}" >&2
  exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]] || [[ ! -f "${MODEL_PATH}/model.safetensors.index.json" ]]; then
  echo "Merged stage-1 Hugging Face model not found: ${MODEL_PATH}" >&2
  echo "Run merge_fsdp_checkpoint.sh on ${DEFAULT_STAGE1_CHECKPOINT} first." >&2
  exit 2
fi

exec "${TOOLEQA_ROOT}/src/train/RFT/scripts/run_evidence_grpo.sh" \
  reward.custom_reward_function.reward_kwargs.reward_phase=joint \
  reward.custom_reward_function.reward_kwargs.no_progress=-0.01 \
  reward.custom_reward_function.reward_kwargs.tool_cost=-0.002 \
  reward.custom_reward_function.reward_kwargs.path_excess=-0.05 \
  algorithm.gdpo_reward_weights='[1.0,0.5]' \
  data.shuffle=true \
  data.train_batch_size=1 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  trainer.total_training_steps=100 \
  trainer.total_epochs=1 \
  trainer.val_before_train=true \
  trainer.save_freq=25 \
  trainer.test_freq=25 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.resume_mode=disable \
  "$@"
