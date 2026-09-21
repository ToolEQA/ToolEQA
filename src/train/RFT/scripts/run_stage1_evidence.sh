#!/usr/bin/env bash
set -euo pipefail

TOOLEQA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-eqa-rt-rft-instruct-thought-code-stage1-balanced450}"
export TRAIN_FILE="${TRAIN_FILE:-${TOOLEQA_ROOT}/src/train/RFT/data/train_staged_reward_eligible_450.jsonl}"
export VAL_FILE="${VAL_FILE:-${TOOLEQA_ROOT}/src/train/RFT/data/validation_stratified_reward_eligible_45.jsonl}"

exec "${TOOLEQA_ROOT}/src/train/RFT/scripts/run_evidence_grpo.sh" \
  reward.custom_reward_function.reward_kwargs.reward_phase=evidence \
  reward.custom_reward_function.reward_kwargs.no_progress=0.0 \
  reward.custom_reward_function.reward_kwargs.tool_cost=0.0 \
  reward.custom_reward_function.reward_kwargs.path_excess=0.0 \
  algorithm.gdpo_reward_weights='[1.0,0.0]' \
  data.shuffle=false \
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
