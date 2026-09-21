#!/usr/bin/env bash
# Isolated short throughput test. This is NOT the formal 450+450-step run.
set -euo pipefail
cd /home/zml/algorithm/ToolEQA
export PYTHON_BIN=/home/zml/programs/miniconda3/envs/verl-tooleqa/bin/python
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export RFT_OUTPUT_ROOT=/mynvme1/ToolEQA_ICLR2027/speed-tests-20260916
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-nccl-two-workers}"
export MODEL_PATH=/mynvme0/models/Qwen/Qwen3-VL-8B-Instruct
export TRAIN_FILE=/mynvme1/ToolEQA_ICLR2027/rft8b-open-20260916/data-v2/train.jsonl
export VAL_FILE=/mynvme1/ToolEQA_ICLR2027/rft8b-open-20260916/data-v2/dev.jsonl
export RAY_VISIBLE_GPUS=0,1,2,3 TRAIN_GPUS=3 ROLLOUT_GPUS=1
export TOOLEQA_TOOL_GPU_ID=0 TOOLEQA_AGENT_GPU_ID=3
export TOOLEQA_DISTRIBUTED_BACKEND=nccl CHECKPOINT_BACKEND=gloo
export VERL_FSDP_WEIGHT_SYNC_CPU=1 VERL_WEIGHT_TRANSFER_SHM=1
export TOOLEQA_LIMIT_NVML_TO_VISIBLE=1 TOOLEQA_NVML_INDEX_MAP=4:6
export TOOLEQA_RANK_DEVICE_MAP=1 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export TOOLEQA_SCENE_ROOT=/data/zml/datasets/EmbodiedQA/HM3D
export TOOLEQA_FROZEN_SERVICE=http://127.0.0.1:18941 PYTHONUNBUFFERED=1
export TOOLEQA_NVML_HEALTHY_PREFIX=4
export LD_PRELOAD="${RFT_OUTPUT_ROOT}/nvml_healthy_prefix.so"
exec bash src/train/RFT/scripts/run_stage1_evidence.sh \
  trainer.total_training_steps=3 trainer.test_freq=-1 trainer.save_freq=-1 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.62 \
  "$@"
