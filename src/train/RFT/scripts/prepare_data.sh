#!/usr/bin/env bash
set -euo pipefail

TOOLEQA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zml/programs/miniconda3/envs/verl-tooleqa/bin/python}"
export TOOLEQA_ROOT
export PYTHONPATH="${TOOLEQA_ROOT}:${TOOLEQA_ROOT}/third_party/verl${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" -m src.train.RFT.prepare_dataset "$@"
