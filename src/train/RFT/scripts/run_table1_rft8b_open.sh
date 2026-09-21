#!/usr/bin/env bash
set -euo pipefail
cd /home/zml/algorithm/ToolEQA
export PYTHON_BIN=/home/zml/programs/miniconda3/envs/verl-tooleqa/bin/python
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export RFT_OUTPUT_ROOT="${RFT_OUTPUT_ROOT:-/mynvme1/ToolEQA_ICLR2027/rft8b-open-memsafe-20260916}"
DATA_ROOT=/mynvme1/ToolEQA_ICLR2027/rft8b-open-20260916/data-v2
export TRAIN_FILE="${DATA_ROOT}/train.jsonl"
export VAL_FILE="${DATA_ROOT}/dev.jsonl"
mkdir -p "${RFT_OUTPUT_ROOT}"
if [[ "${TOOLEQA_SUPERVISED:-0}" != "1" ]]; then
  exec "${PYTHON_BIN}" -m src.train.RFT.supervise_run \
    --directory "${RFT_MONITOR_DIR:-${RFT_OUTPUT_ROOT}/monitor}" --progress-root "${RFT_OUTPUT_ROOT}" \
    -- env TOOLEQA_SUPERVISED=1 bash "${BASH_SOURCE[0]}"
fi
if [[ ! -e "${RFT_OUTPUT_ROOT}/data-v2" ]]; then
  ln -s "${DATA_ROOT}" "${RFT_OUTPUT_ROOT}/data-v2"
fi
export RAY_VISIBLE_GPUS=0,1,2,3
export TRAIN_GPUS=3 ROLLOUT_GPUS=1
export TOOLEQA_TOOL_GPU_ID=0 TOOLEQA_AGENT_GPU_ID=3
export TOOLEQA_DISTRIBUTED_BACKEND=nccl CHECKPOINT_BACKEND=gloo
export VERL_FSDP_WEIGHT_SYNC_CPU=1 VERL_WEIGHT_TRANSFER_SHM=1
export TOOLEQA_LIMIT_NVML_TO_VISIBLE=1 TOOLEQA_NVML_INDEX_MAP=4:6
export TOOLEQA_RANK_DEVICE_MAP=1 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export TOOLEQA_SCENE_ROOT=/data/zml/datasets/EmbodiedQA/HM3D
export TOOLEQA_FROZEN_SERVICE=http://127.0.0.1:18941
export PYTHONUNBUFFERED=1
unset TOOLEQA_STRESS_LONGEST_TURN
exec 9>"${RFT_OUTPUT_ROOT}/pipeline.lock"
flock -n 9 || { echo 'Table 1 open pipeline already running'; exit 2; }
"${PYTHON_BIN}" src/train/RFT/scripts/check_five_gpus.py
"${PYTHON_BIN}" -c 'from src.train.RFT.open_protocol import semantic_judgment; assert semantic_judgment("What color is the chair?", "Red.", "The chair is red.")["score"] == 5'
export TOOLEQA_NVML_HEALTHY_PREFIX=4 TOOLEQA_FAST_PATCH_EMBED=1
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
export TOOLEQA_TRAIN_EXPANDABLE_SEGMENTS=1
export TOOLEQA_MEMORY_AUDIT=1
export TOOLEQA_EARLY_CHECKPOINT_STEPS="${TOOLEQA_EARLY_CHECKPOINT_STEPS:-1,5}"
export RAY_DEDUP_LOGS_ALLOW_REGEX='TOOLEQA_MEMORY|TOOLEQA_ALLOCATOR'
export LD_PRELOAD=/mynvme1/ToolEQA_ICLR2027/speed-tests-20260916/nvml_healthy_prefix.so
test -r "${LD_PRELOAD}"
COMMON=(trainer.test_freq=150 trainer.save_freq=150 trainer.max_actor_ckpt_to_keep=5
  actor_rollout_ref.actor.use_dynamic_bsz=false
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.agent.num_workers=2
  "actor_rollout_ref.rollout.gpu_memory_utilization=${RFT_ROLLOUT_MEMORY_UTILIZATION:-0.58}")
if [[ "${RFT_RESUME_STAGE:-1}" == "2" ]]; then
  [[ "${RFT_RESUME_STEP:-}" =~ ^[0-9]+$ ]] && (( RFT_RESUME_STEP > 0 && RFT_RESUME_STEP < 450 )) || exit 2
  RESUME_PATH="${RFT_OUTPUT_ROOT}/stage2-joint/checkpoints/global_step_${RFT_RESUME_STEP}"
  test -s "${RESUME_PATH}/data.pt"
  for rank in 0 1 2; do
    for kind in model optim extra_state; do
      test -s "${RESUME_PATH}/actor/${kind}_world_size_3_rank_${rank}.pt"
    done
  done
  export MODEL_PATH="${RFT_OUTPUT_ROOT}/stage1-evidence/checkpoints/global_step_450/actor_huggingface"
  export EXPERIMENT_NAME=stage2-joint TOOLEQA_RESUME_REWIND_PREFETCH=1
  echo "$(date -Is) RESUME stage2 step${RFT_RESUME_STEP}, rollout_memory=${RFT_ROLLOUT_MEMORY_UTILIZATION:-0.58}" | tee -a "${RFT_OUTPUT_ROOT}/stage2.log"
  bash src/train/RFT/scripts/run_stage2_joint.sh "${COMMON[@]}" \
    trainer.resume_mode=resume_path "trainer.resume_from_path=${RESUME_PATH}" >> "${RFT_OUTPUT_ROOT}/stage2.log" 2>&1
  echo "$(date -Is) TRAINING COMPLETE: stage2 step450."
  exit 0
fi
export MODEL_PATH=/mynvme0/models/Qwen/Qwen3-VL-8B-Instruct
export EXPERIMENT_NAME=stage1-evidence
RESUME_ARGS=()
if [[ -n "${RFT_RESUME_STEP:-}" ]]; then
  [[ "${RFT_RESUME_STEP}" =~ ^[0-9]+$ ]] && (( RFT_RESUME_STEP > 0 && RFT_RESUME_STEP < 450 )) || exit 2
  RESUME_PATH="${RFT_OUTPUT_ROOT}/stage1-evidence/checkpoints/global_step_${RFT_RESUME_STEP}"
  test -s "${RESUME_PATH}/data.pt"
  for rank in 0 1 2; do
    for kind in model optim extra_state; do
      test -s "${RESUME_PATH}/actor/${kind}_world_size_3_rank_${rank}.pt"
    done
  done
  [[ ! -e "${RFT_OUTPUT_ROOT}/stage2-joint" ]] || exit 2
  RESUME_ARGS=(trainer.resume_mode=resume_path "trainer.resume_from_path=${RESUME_PATH}")
  export TOOLEQA_RESUME_REWIND_PREFETCH=1
elif [[ -e "${RFT_OUTPUT_ROOT}/${EXPERIMENT_NAME}" || -e "${RFT_OUTPUT_ROOT}/stage1.log" ]]; then
  echo 'Existing experiment artifacts; refusing implicit overwrite/restart.'
  exit 2
fi
echo "$(date -Is) START stage1: resume=${RFT_RESUME_STEP:-disabled}, open answers, target450 steps, n=6" | tee -a "${RFT_OUTPUT_ROOT}/stage1.log"
bash src/train/RFT/scripts/run_stage1_evidence.sh "${COMMON[@]}" "${RESUME_ARGS[@]}" >> "${RFT_OUTPUT_ROOT}/stage1.log" 2>&1
echo "$(date -Is) MERGE stage1 step450"
unset TOOLEQA_RESUME_REWIND_PREFETCH
STAGE1="${RFT_OUTPUT_ROOT}/stage1-evidence/checkpoints/global_step_450"
bash src/train/RFT/scripts/merge_fsdp_checkpoint.sh "${STAGE1}" "${STAGE1}/actor_huggingface" > "${RFT_OUTPUT_ROOT}/merge-stage1.log" 2>&1
export MODEL_PATH="${STAGE1}/actor_huggingface"
export EXPERIMENT_NAME=stage2-joint
echo "$(date -Is) START stage2: open semantic reward, 450 steps, n=6"
bash src/train/RFT/scripts/run_stage2_joint.sh "${COMMON[@]}" > "${RFT_OUTPUT_ROOT}/stage2.log" 2>&1
echo "$(date -Is) TRAINING COMPLETE: select checkpoint by 225-task dev LLM-Match before full Seen/Unseen."
