#!/usr/bin/env bash
set -euo pipefail

TOOLEQA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DETANY_PYTHON="${DETANY_PYTHON:-/home/zml/programs/miniconda3/envs/detany3d/bin/python}"
DETANY_GPU="${DETANY_GPU:-3}"
TOOLEQA_TOOL_GPU_ID="${TOOLEQA_TOOL_GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="${DETANY_GPU}"
export PYTHONPATH="${TOOLEQA_ROOT}:${TOOLEQA_ROOT}/third_party/DetAny3D${PYTHONPATH:+:${PYTHONPATH}}"

echo "DetAny3D: physical GPU ${DETANY_GPU}, shared-memory channel ${TOOLEQA_TOOL_GPU_ID}"
exec "${DETANY_PYTHON}" "${TOOLEQA_ROOT}/src/train/RFT/detany_server.py" --channel "${TOOLEQA_TOOL_GPU_ID}" "$@"
