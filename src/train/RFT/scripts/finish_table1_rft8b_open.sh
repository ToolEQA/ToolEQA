#!/usr/bin/env bash
# Compatibility launcher. Evaluation orchestration now lives in Python.
set -euo pipefail
cd /home/zml/algorithm/ToolEQA
PYTHON_BIN="${PYTHON_BIN:-/home/zml/programs/miniconda3/envs/verl-tooleqa/bin/python}"
RFT_OUTPUT_ROOT="${RFT_OUTPUT_ROOT:-/mynvme1/ToolEQA_ICLR2027/rft8b-open-memsafe-20260916}"
exec "${PYTHON_BIN}" -m src.evaluation.run_open_eval --run-root "${RFT_OUTPUT_ROOT}" --execute "$@"
