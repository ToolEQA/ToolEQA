# ToolEQA verl Adapter

This directory contains the first-pass integration layer for training the ToolEQA controller with `verl`.

## Scope

- Keep planner fixed
- Keep executor/tool runtime fixed
- Train only the controller
- Use verl for GRPO/PPO, multi-turn rollout, tool-call orchestration, and reward computation

## Main files

- `dataset_builder.py`: convert ToolEQA dataset to verl-friendly jsonl
- `env_bridge.py`: bridge ToolEQA tools/environment into a multi-turn runtime
- `tool_wrappers.py`: verl-style tool wrappers
- `agent_loop.py`: user-defined agent loop for ToolEQA controller rollout
- `reward_manager.py`: function-based reward implementation
- `configs/grpo_tooleqa.yaml`: recommended config for Qwen2.5-VL style controller training
- `configs/ppo_tooleqa.yaml`: reference PPO config; only valid for models with a compatible value-head/critic path

## Current status

This is an integration scaffold. It is designed to minimize changes to the existing ToolEQA runtime and make the next step practical:

1. prepare dataset
2. run a minimal multi-turn rollout
3. wire the rollout into a real verl GRPO/PPO job

Depending on the installed `verl` version, the config keys or import paths may need small adjustments before the first full training run.

## Notes

- For multimodal controller models such as `Qwen2.5-VL`, use `grpo_tooleqa.yaml` by default.
- The local non-Docker install in this repo currently includes `vllm`, not `sglang`, so the default rollout backend is `vllm`.
- `ppo_tooleqa.yaml` will fail on plain VL checkpoints unless you provide a critic-compatible value-head model.
- The current `grpo_tooleqa.yaml` enables FSDP parameter and optimizer offload by default to make single-GPU dry-runs more practical on `Qwen2.5-VL` class models.
- The local Python 3.12 `verl` environment currently needs startup compatibility patches for `Qwen2Tokenizer.all_special_tokens_extended` and `Qwen2VLImageProcessor.min_pixels/max_pixels` so that `vllm 0.11.0` can boot against the installed `transformers` stack.
- A patched local test model directory at `/tmp/qwen25vl3b_vllm_patched` was used during bring-up to replace legacy `rope_scaling.type=mrope` with the newer `rope_scaling.rope_type=mrope` form expected by `vllm`.


bash -lc 
'
export PATH=/tmp/verl-py312/bin:$PATH; 
export CUDA_VISIBLE_DEVICES=0; 
python -m verl.trainer.main_ppo --config-path=/home/zml/algorithm/ToolEQA/src/train/RFT/verl_adapter/configs --config-name=grpo_tooleqa actor_rollout_ref.model.path=/tmp/qwen25vl3b_vllm_patched actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true actor_rollout_ref.ref.fsdp_config.param_offload=true actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.05 actor_rollout_ref.rollout.max_num_batched_tokens=512 actor_rollout_ref.rollout.max_num_seqs=2 trainer.val_only=true trainer.n_gpus_per_node=1 trainer.nnodes=1 data.train_files=[src/train/RFT/verl_adapter/data/tooleqa_train.sample.jsonl] data.val_files=[src/train/RFT/verl_adapter/data/tooleqa_train.sample.jsonl]
'