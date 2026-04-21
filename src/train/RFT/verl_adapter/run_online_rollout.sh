#!/bin/bash
# Online RL training with real Habitat rollout.
#
# GPU layout (4 GPUs):
#   GPU 0-1: vllm (rollout generation) + Habitat environment (shared, memory-partitioned)
#   GPU 2-3: FSDP actor + reference model (with param/optimizer offload)
#
# Memory strategy:
#   vllm gpu_memory_utilization=0.4 leaves ~60% GPU memory for Habitat simulator.
#   If OOM occurs, reduce to 0.3 or increase number of GPUs.

set -e

# Silence Habitat and Magnum logs
export HABITAT_SIM_LOG="quiet"
export MAGNUM_LOG="quiet"
export PYTHONWARNINGS="ignore"

# Use the verl-py312 environment
export PATH=/tmp/verl-py312/bin:$PATH

CONFIG_PATH="$(cd "$(dirname "$0")" && pwd)/configs"

echo "Launching ToolEQA online RL training..."
echo "Config path: $CONFIG_PATH"
echo "Data: src/train/RFT/verl_adapter/data/tooleqa_train_with_neg.jsonl"
echo "GPUs: 4 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name=grpo_tooleqa_online \
    trainer.n_gpus_per_node=4 \
    actor_rollout_ref.rollout.engine_kwargs.vllm.gpu_memory_utilization=0.4
