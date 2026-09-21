#!/usr/bin/env bash
set -euo pipefail

TOOLEQA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DEFAULT_STAGE1_MODEL="/mynvme0/ToolEQA_RFT/eqa-rt-rft-v6-evidence-stage1-balanced450/checkpoints/global_step_450/actor_huggingface"
export MODEL_PATH="${MODEL_PATH:-${DEFAULT_STAGE1_MODEL}}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-eqa-rt-rft-v6-joint-stage2-balanced450}"
export TRAIN_FILE="${TRAIN_FILE:-${TOOLEQA_ROOT}/src/train/RFT/data/train_balanced_reward_eligible_450.jsonl}"
export VAL_FILE="${VAL_FILE:-${TOOLEQA_ROOT}/src/train/RFT/data/validation_stratified_reward_eligible_45.jsonl}"

if [[ ! -f "${MODEL_PATH}/config.json" ]] || [[ ! -f "${MODEL_PATH}/model.safetensors.index.json" ]]; then
  echo "Merged stage-1 Hugging Face model not found: ${MODEL_PATH}" >&2
  echo "Run merge_fsdp_checkpoint.sh on the stage-1 global_step_450 checkpoint first." >&2
  exit 2
fi

exec "${TOOLEQA_ROOT}/src/train/RFT/scripts/run_evidence_grpo.sh" \
  reward.custom_reward_function.reward_kwargs.reward_phase=joint \
  reward.custom_reward_function.reward_kwargs.no_progress=-0.01 \
  reward.custom_reward_function.reward_kwargs.tool_cost=-0.002 \
  reward.custom_reward_function.reward_kwargs.path_excess=-0.05 \
  algorithm.gdpo_reward_weights='[1.0,1.0]' \
  data.shuffle=true \
  data.train_batch_size=1 \
  actor_rollout_ref.rollout.n=6 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  trainer.total_training_steps=450 \
  trainer.total_epochs=1 \
  trainer.val_before_train=false \
  trainer.save_freq=150 \
  trainer.test_freq=50 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.resume_mode=disable \
  "$@"
