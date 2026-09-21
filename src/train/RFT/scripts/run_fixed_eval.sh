#!/usr/bin/env bash
set -euo pipefail

TOOLEQA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MODE="${1:-base}"
if [[ $# -gt 0 ]]; then
  shift
fi
VAL_FILE="${VAL_FILE:-${TOOLEQA_ROOT}/src/train/RFT/data/validation_reward_eligible.jsonl}"

case "${MODE}" in
  base)
    export EXPERIMENT_NAME="${EXPERIMENT_NAME:-fixed-eval-base-qwen3-vl-8b-instruct-thought-code}"
    RESUME_ARGS=(trainer.resume_mode=disable)
    ;;
  checkpoint)
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mynvme0/ToolEQA_RFT/eqa-rt-rft-instruct-thought-code-stage1-balanced450/checkpoints/global_step_450}"
    MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-${CHECKPOINT_PATH}/actor_huggingface}"
    if [[ ! -f "${MERGED_MODEL_PATH}/config.json" ]]; then
      echo "Merged checkpoint model not found: ${MERGED_MODEL_PATH}" >&2
      echo "Run src/train/RFT/scripts/merge_fsdp_checkpoint.sh ${CHECKPOINT_PATH} first." >&2
      exit 2
    fi
    export MODEL_PATH="${MERGED_MODEL_PATH}"
    export EXPERIMENT_NAME="${EXPERIMENT_NAME:-fixed-eval-instruct-thought-code-stage1-balanced450}"
    RESUME_ARGS=(trainer.resume_mode=disable)
    ;;
  *)
    echo "Usage: $0 [base|checkpoint] [Hydra overrides ...]" >&2
    exit 2
    ;;
esac

exec "${TOOLEQA_ROOT}/src/train/RFT/scripts/run_evidence_grpo.sh" \
  "${RESUME_ARGS[@]}" \
  "data.val_files=[${VAL_FILE}]" \
  data.val_max_samples=-1 \
  data.shuffle=false \
  data.validation_shuffle=false \
  trainer.val_before_train=true \
  trainer.val_only=true \
  trainer.test_freq=-1 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  "$@"
