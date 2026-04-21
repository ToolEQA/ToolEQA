# ToolEQA RFT

这个目录用于放 `ToolEQA controller` 的强化微调相关设计、适配代码、启动脚本和说明。

## 目录结构

- [强化.pdf](/home/zml/algorithm/ToolEQA/src/train/RFT/强化.pdf): 设计文档
- [verl_controller_rft_plan.md](/home/zml/algorithm/ToolEQA/src/train/RFT/verl_controller_rft_plan.md): 基于设计文档整理的执行计划
- [scripts/run_grpo_train.sh](/home/zml/algorithm/ToolEQA/src/train/RFT/scripts/run_grpo_train.sh): 当前推荐的训练启动脚本
- [verl_adapter](/home/zml/algorithm/ToolEQA/src/train/RFT/verl_adapter): `ToolEQA` 接入 `verl` 的适配层

## 当前方案

当前不是直接把原来的 `ReactCodeAgent` 拿去做 RL，而是把 `controller` 适配成 `verl` 的多轮 tool-calling 任务：

- `ToolEQA` 继续负责环境、工具和 Habitat 执行
- `verl` 负责 rollout、GRPO 优势计算、策略更新和多轮 agent loop
- 强化训练对象只包括 `controller`
- `planner` 和底层工具执行逻辑保持固定

## 当前训练入口

当前默认训练入口是：

```bash
python -m verl.experimental.one_step_off_policy.main_ppo
```

虽然入口文件名里有 `ppo`，但当前实际算法是 `GRPO`，由配置里的：

```yaml
algorithm:
  adv_estimator: grpo
```

控制。

## 为什么用 One-Step-Off

当前配置已经改成“训练卡 / rollout 卡分离”。

原因是 `Qwen2.5-VL + FSDP actor/ref + vLLM rollout` 在单卡共置下很容易 OOM。现在默认走 `one_step_off_policy`：

- 训练卡负责 `actor/ref`
- rollout 卡负责 `vLLM rollout`
- 不再让训练和 rollout 抢同一张 GPU

对应配置在 [grpo_tooleqa.yaml](/home/zml/algorithm/ToolEQA/src/train/RFT/verl_adapter/configs/grpo_tooleqa.yaml)：

- `actor_rollout_ref.hybrid_engine: false`
- 顶层 `rollout.n_gpus_per_node: 1`
- `trainer.n_gpus_per_node: 1`

## 当前 reward

当前训练实际使用的是 [reward_fn.py](/home/zml/algorithm/ToolEQA/src/train/RFT/verl_adapter/reward_fn.py)。

它已经并入了 [reward_manager.py](/home/zml/algorithm/ToolEQA/src/train/RFT/verl_adapter/reward_manager.py) 的过程奖励，当前有效项包括：

- `r_ans`: 最终答案正确 `+1.0`，错误 `-1.0`
- `r_redund`: 重复动作惩罚 `-0.05`
- `r_prem`: 没找全相关对象就提前回答，惩罚 `-0.3`
- `r_find`: 首次找到相关对象，奖励 `+0.2`
- `r_info`: `ObjectCrop` / `VisualQA` 信息收集奖励 `+0.1`

负样本逻辑也保留了：

- `wrong_answer` 负样本命中错误答案时给 `0.3`
- 其他负样本给 `0.0`

## 最小启动方式

默认脚本：

```bash
bash src/train/RFT/scripts/run_grpo_train.sh
```

当前脚本默认会使用：

- `PYTHON_BIN=/tmp/verl-py312/bin/python`
- `CUDA_VISIBLE_DEVICES=0,1`
- `TRAIN_GPUS_PER_NODE=1`
- `ROLLOUT_GPUS_PER_NODE=1`

也就是默认要求至少 2 张可见 GPU。

如果只想先检查命令而不真正启动训练：

```bash
DRY_RUN=true bash src/train/RFT/scripts/run_grpo_train.sh
```

## 常用启动示例

双卡分离启动：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_GPUS_PER_NODE=1 \
ROLLOUT_GPUS_PER_NODE=1 \
bash src/train/RFT/scripts/run_grpo_train.sh
```

指定在线训练配置：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
CONFIG_NAME=grpo_tooleqa_online \
TRAIN_GPUS_PER_NODE=1 \
ROLLOUT_GPUS_PER_NODE=1 \
VAL_ONLY=false \
bash src/train/RFT/scripts/run_grpo_train.sh
```

## 关键文件

- [verl_adapter/env_bridge.py](/home/zml/algorithm/ToolEQA/src/train/RFT/verl_adapter/env_bridge.py): ToolEQA 环境桥接
- [verl_adapter/tool_wrappers.py](/home/zml/algorithm/ToolEQA/src/train/RFT/verl_adapter/tool_wrappers.py): 工具包装
- [verl_adapter/verl_tool_agent_loop.py](/home/zml/algorithm/ToolEQA/src/train/RFT/verl_adapter/verl_tool_agent_loop.py): 自定义 agent loop
- [verl_adapter/reward_fn.py](/home/zml/algorithm/ToolEQA/src/train/RFT/verl_adapter/reward_fn.py): 当前生效的 reward
- [verl_adapter/configs/grpo_tooleqa.yaml](/home/zml/algorithm/ToolEQA/src/train/RFT/verl_adapter/configs/grpo_tooleqa.yaml): 默认 GRPO 配置
- [verl_adapter/configs/grpo_tooleqa_online.yaml](/home/zml/algorithm/ToolEQA/src/train/RFT/verl_adapter/configs/grpo_tooleqa_online.yaml): 在线训练配置

## 环境说明

当前本地 `verl` 安装和源码位置：

- Python 环境：`/tmp/verl-py312`
- `verl` 源码：`third_party/verl`

脚本不会修改你原来的 `react-eqa` 环境，而是默认使用这个独立的 Python 3.12 环境。

## 备注

- 当前 `run_grpo_train.sh` 会在可见 GPU 数不足时直接报错退出，避免再次落回“训练和 rollout 共置一张卡导致 OOM”的情况。
- `verl_adapter/README.md` 记录的是更细的 adapter 说明；本 README 主要面向这个目录的整体使用。
