#!/usr/bin/env bash
set -euo pipefail
cd /home/zml/algorithm/ToolEQA
RUN_ROOT=/mynvme0/ToolEQA_RFT/step0-official-tests-20260914
PYTHON_BIN=/home/zml/programs/miniconda3/envs/verl-tooleqa/bin/python
# Never equate nvidia-smi visibility with CUDA usability.
"${PYTHON_BIN}" src/train/RFT/scripts/check_five_gpus.py
if [[ ! -d /data/zml/datasets/EmbodiedQA/HM3D ]]; then
  echo 'HM3D mount is missing: /data/zml/datasets/EmbodiedQA/HM3D; evaluation remains stopped.' >&2
  exit 2
fi
if pgrep -f '^bash src/train/RFT/scripts/run_step0_official_tests.sh$' > /dev/null; then
  echo 'The evaluation launcher is already running.' >&2
  exit 2
fi
if pgrep -f '^/home/zml/programs/miniconda3/envs/detany3d/bin/python /home/zml/algorithm/ToolEQA/src/train/RFT/detany_server.py --channel 0$' > /dev/null; then
  echo 'An existing DetAny channel 0 process needs inspection before restart.' >&2
  exit 2
fi
nohup setsid env DETANY_GPU=GPU-dd87e394-2826-1441-19c5-e01befc9c8e3 PYTHONUNBUFFERED=1 \
  bash src/train/RFT/scripts/run_detany3d.sh >> "${RUN_ROOT}/detany-recovery.log" 2>&1 < /dev/null &
service_pid=$!
ready=0
for ((attempt=0; attempt<90; attempt++)); do
  if ! kill -0 "${service_pid}" 2>/dev/null; then
    echo 'DetAny exited during initialization; inspect detany-recovery.log.' >&2
    exit 1
  fi
  if [[ -e /dev/shm/image_data_0 && -e /dev/shm/result_data_0 ]]; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" != 1 ]]; then
  echo 'DetAny initialization did not complete; evaluation was not launched.' >&2
  exit 1
fi
nohup setsid bash src/train/RFT/scripts/run_step0_official_tests.sh \
  >> "${RUN_ROOT}/launcher.log" 2>&1 < /dev/null &
evaluation_pid=$!
sleep 2
if ! kill -0 "${evaluation_pid}" 2>/dev/null; then
  echo 'Evaluation launcher exited; inspect launcher.log.' >&2
  exit 1
fi
nohup setsid "${PYTHON_BIN}" -u -m src.train.RFT.paper_followup \
  --run-root "${RUN_ROOT}" --session-id 01a09dcf-bc15-7a51-ae80-6903fc1bbd45 \
  >> "${RUN_ROOT}/paper-followup-watch.log" 2>&1 < /dev/null &
echo "Evaluation PID: ${evaluation_pid}; DetAny startup PID: ${service_pid}"
