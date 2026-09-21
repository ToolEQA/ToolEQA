#!/usr/bin/env bash
set -euo pipefail
cd /home/zml/algorithm/ToolEQA
PYTHON_BIN=/home/zml/programs/miniconda3/envs/verl-tooleqa/bin/python
export MERGED_MODEL_PATH=/mynvme0/ToolEQA_RFT/eqa-rt-rft-instruct-thought-code-stage1-pilot150/checkpoints/global_step_150/actor_huggingface
export OFFICIAL_TEST=1
export RAY_VISIBLE_GPUS=0,1,2,3
export TOOLEQA_AGENT_GPU_ID=3
export TOOLEQA_TOOL_GPU_ID=0
export TOOLEQA_LIMIT_NVML_TO_VISIBLE=1
export TOOLEQA_NVML_INDEX_MAP=4:6
export TOOLEQA_DISTRIBUTED_BACKEND=gloo
export CHECKPOINT_BACKEND=gloo
export VERL_FSDP_WEIGHT_SYNC_CPU=1
export VERL_WEIGHT_TRANSFER_SHM=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export TOOLEQA_RANK_DEVICE_MAP=1
export PYTHONUNBUFFERED=1
export TOOLEQA_SCENE_ROOT=/data/zml/datasets/EmbodiedQA/HM3D
RUN_ROOT=/mynvme0/ToolEQA_RFT/step0-official-tests-20260914
echo "$$" > "${RUN_ROOT}/evaluation.pid"
trap 'code=$?; date -Is; echo "evaluation_exit_code=${code}"' EXIT
"${PYTHON_BIN}" -m src.train.RFT.resume_official prepare --run-root "${RUN_ROOT}"
for split in seen unseen; do
  echo "$(date -Is) Starting ${split}"
  export VAL_FILE="${RUN_ROOT}/data-resume/${split}.jsonl"
  export EXPERIMENT_NAME="step0-official-tests-20260914/${split}"
  if [[ -s "${VAL_FILE}" ]]; then
    bash src/train/RFT/scripts/run_fixed_eval.sh checkpoint \
      reward.custom_reward_function.path=/home/zml/algorithm/ToolEQA/src/train/RFT/official_eval.py \
      >> "${RUN_ROOT}/${split}.log" 2>&1
  fi
  "${PYTHON_BIN}" -m src.train.RFT.resume_official consolidate --run-root "${RUN_ROOT}" --split "${split}"
  echo "$(date -Is) Completed ${split}"
done
