#!/usr/bin/env bash
# Diagnostic run only: starts with the task corresponding to failed update6.
set -euo pipefail
cd /home/zml/algorithm/ToolEQA
export EXPERIMENT_NAME=memory-safe-longest-v2
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
export TOOLEQA_TRAIN_EXPANDABLE_SEGMENTS=1
export TOOLEQA_FAST_PATCH_EMBED=1 TOOLEQA_MEMORY_AUDIT=1
export RAY_DEDUP_LOGS_ALLOW_REGEX='TOOLEQA_MEMORY|TOOLEQA_ALLOCATOR'
export TOOLEQA_STRESS_LONGEST_TURN=1 TOOLEQA_EARLY_CHECKPOINT_STEPS=1
exec bash src/train/RFT/scripts/run_speed_probe.sh \
  data.train_files='[/mynvme1/ToolEQA_ICLR2027/memory-tests-20260916/stress-train.jsonl]' \
  trainer.total_training_steps=3 trainer.save_freq=3 trainer.max_actor_ckpt_to_keep=2 \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
