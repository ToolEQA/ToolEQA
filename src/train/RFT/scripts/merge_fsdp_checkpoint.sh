#!/usr/bin/env bash
set -euo pipefail

TOOLEQA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zml/programs/miniconda3/envs/verl-tooleqa/bin/python}"
CHECKPOINT_PATH="${1:-/mynvme0/ToolEQA_RFT/eqa-rt-rft-instruct-thought-code-stage1-balanced450/checkpoints/global_step_450}"
TARGET_DIR="${2:-${CHECKPOINT_PATH}/actor_huggingface}"
ACTOR_DIR="${CHECKPOINT_PATH}/actor"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mynvme0/models/Qwen/Qwen3-VL-8B-Instruct}"

if [[ ! -f "${ACTOR_DIR}/fsdp_config.json" ]]; then
  echo "FSDP actor checkpoint not found: ${ACTOR_DIR}" >&2
  exit 2
fi
if [[ -e "${TARGET_DIR}" ]]; then
  echo "Target already exists; refusing to overwrite: ${TARGET_DIR}" >&2
  exit 2
fi
if [[ ! -f "${BASE_MODEL_PATH}/config.json" ]] || [[ ! -f "${BASE_MODEL_PATH}/tokenizer.json" ]]; then
  echo "Base model tokenizer/config assets not found: ${BASE_MODEL_PATH}" >&2
  exit 2
fi

export PYTHONPATH="${TOOLEQA_ROOT}/third_party/verl${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${TARGET_DIR}"

# The merger reserializes tokenizer/config assets and, with Transformers 4.57,
# can change transformers_version in a way that falsely activates the Mistral
# regex migration for a Qwen tokenizer. Evaluation must differ from the base
# model only in actor weights, so restore these files byte-for-byte.
ASSET_BACKUP="${TARGET_DIR}/merger_generated_assets_backup"
mkdir -p "${ASSET_BACKUP}"
for name in config.json generation_config.json merges.txt preprocessor_config.json \
  tokenizer.json tokenizer_config.json video_preprocessor_config.json vocab.json; do
  if [[ -f "${TARGET_DIR}/${name}" ]]; then
    mv "${TARGET_DIR}/${name}" "${ASSET_BACKUP}/${name}"
  fi
  cp -a "${BASE_MODEL_PATH}/${name}" "${TARGET_DIR}/${name}"
done
if [[ -f "${BASE_MODEL_PATH}/chat_template.json" ]]; then
  cp -a "${BASE_MODEL_PATH}/chat_template.json" "${TARGET_DIR}/chat_template.json"
fi
