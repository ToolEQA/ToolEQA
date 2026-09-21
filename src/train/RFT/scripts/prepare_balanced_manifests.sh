#!/usr/bin/env bash
set -euo pipefail

TOOLEQA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zml/programs/miniconda3/envs/verl-tooleqa/bin/python}"
DATA_DIR="${DATA_DIR:-${TOOLEQA_ROOT}/src/train/RFT/data}"
export PYTHONPATH="${TOOLEQA_ROOT}:${TOOLEQA_ROOT}/third_party/verl${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m src.train.RFT.select_fixed_validation \
  --source "${DATA_DIR}/train_reward_eligible.jsonl" \
  --output "${DATA_DIR}/train_balanced_reward_eligible_450.jsonl" \
  --stratify-field question_type \
  --per-group 50

"${PYTHON_BIN}" -m src.train.RFT.build_staged_curriculum \
  --input "${DATA_DIR}/train_balanced_reward_eligible_450.jsonl" \
  --output "${DATA_DIR}/train_staged_reward_eligible_450.jsonl" \
  --warmup-per-level 50

"${PYTHON_BIN}" -m src.train.RFT.select_fixed_validation \
  --source "${DATA_DIR}/validation_reward_eligible.jsonl" \
  --output "${DATA_DIR}/validation_stratified_reward_eligible_45.jsonl" \
  --stratify-field question_type \
  --per-group 5
