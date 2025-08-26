#!/bin/bash

# 固定分布式训练配置
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500
NNODES=1
NPROC_PER_NODE=4

# DeepSpeed配置
deepspeed=./scripts/zero3.json

# 模型路径
llm=/mnt/hdd/zml/models/Qwen/Qwen2.5-VL-7B-Instruct/

# 训练超参数
lr=2e-7
batch_size=1
grad_accum_steps=4

# 入口文件
entry_file=qwenvl/train/train_qwen.py

# 数据集
datasets=reacteqa

# 输出配置
run_name="qwen2.5vl-baseline"
output_dir=/mnt/hdd/zml/output

# 训练参数
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --data_packing True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard
"

# 启动训练
torchrun --nnodes=${NNODES} \
         --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}