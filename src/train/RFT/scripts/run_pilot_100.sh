#!/usr/bin/env bash
set -euo pipefail

TOOLEQA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-vl-8b-instruct-thought-code-pilot-100}"
VAL_FILE="${VAL_FILE:-${TOOLEQA_ROOT}/src/train/RFT/data/validation_stratified_reward_eligible_45.jsonl}"

exec "${TOOLEQA_ROOT}/src/train/RFT/scripts/run_evidence_grpo.sh" \
  "data.val_files=[${VAL_FILE}]" \
  data.val_max_samples=-1 \
  data.validation_shuffle=false \
  data.train_batch_size=1 \
  actor_rollout_ref.rollout.n=6 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  trainer.total_training_steps=100 \
  trainer.total_epochs=1 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.resume_mode=disable \
  "$@"
