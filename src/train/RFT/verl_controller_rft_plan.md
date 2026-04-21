# ToolEQA Controller 基于 verl 的强化微调执行计划

## 1. 目标与边界

基于 `src/train/RFT/强化.pdf` 的建议，我的执行目标不是直接对整个 ToolEQA agent 做端到端 RL，而是先做一个更稳、更容易验证收益的版本：

- 固定 `planner`
- 固定 `executor` / tool execution 逻辑
- 只对 `controller` 做强化微调
- 保留 `thought`，但 RL 主目标只优化结构化动作决策
- 训练顺序采用：
  `SFT warm start -> Offline RL -> Online RL`

这个方案与当前仓库结构是对齐的：

- `src/agents/tool_eqa_agent_mp.py` 里的 `EQAReactAgent.step()` 已经天然是逐步决策接口
- `src/tools/*.py` 已经定义了可执行动作空间
- `data/ToolTrajectory/trainval.json` 已经提供问题、计划、轨迹与 react 监督数据
- `src/runs/eqa_modeling.py` 已经提供 Habitat 环境与 rollout 基础

## 2. 总体策略

我会先把现有自由形式 `thought + code` controller，收敛为一个适合 verl/PPO 的“结构化 controller policy”：

- 输入状态：
  `question + planner plan + 当前图像 + 历史轨迹摘要 + evidence memory + 剩余 budget`
- 输出动作：
  `action_type + action_args`
- 动作类型优先固定为：
  `Navigate / ObjectLocation2D / ObjectLocation3D / ObjectCrop / VisualQA / FinalAnswer`

其中：

- `Navigate` 参数限制为 `{move_forward, turn_left, turn_right, stop}` 或映射到当前 `GoNextPointTool`
- `thought` 不作为 RL 的直接 action，而作为 auxiliary loss 或蒸馏目标保留
- 真正进入 PPO/verl 优化的是结构化动作头，而不是整段 Python code

这样可以避免直接对代码生成做 RL，降低 credit assignment 难度，也更符合设计文档里“RL 只优化结构化动作头”的建议。

## 3. 分阶段执行

## Phase 0：接口梳理与最小闭环

目标：先把“当前 controller 在每一步到底看到了什么、做了什么、得到了什么”整理成稳定训练接口。

需要做的事：

1. 抽离 controller step 数据结构
   统一每一步的字段：
   - `obs_image`
   - `question`
   - `plan`
   - `history`
   - `evidence_memory`
   - `action_type`
   - `action_args`
   - `tool_observation`
   - `done`
   - `answer`

2. 明确当前动作到结构化动作的映射
   从 `react.code` 中解析：
   - `GoNextPointTool(...)`
   - `ObjectLocation2D(...)`
   - `ObjectLocation3D(...)`
   - `ObjectCrop(...)`
   - `VisualQATool(...)`
   - `final_answer(...)`

3. 定义最小版本 evidence memory
   第一版不求完备，只记录：
   - 已发现对象集合
   - 已调用工具集合
   - 最近一次 crop / vqa 结果
   - 是否已具备回答候选证据
   - 历史 FinalAnswer 尝试次数

交付物建议：

- `src/train/RFT/action_schema.py`
- `src/train/RFT/parse_react_trajectory.py`
- `src/train/RFT/evidence_memory.py`

## Phase 1：SFT warm start 数据整理

目标：先训练一个“会输出结构化动作”的 controller 初始策略，再上 RL。

数据来源：

- 主数据：`data/ToolTrajectory/trainval.json`
- 可选补充：已有运行结果 `result_*.jsonl` 中的 `summary.react`

具体做法：

1. 将 `trainval.json` 转成 step-level SFT 样本
   每个 step 生成一条样本：
   - 输入：`question + plan + current image + history summary`
   - 输出：`action_type + action_args`

2. 同步生成 auxiliary thought target
   - 结构化动作是主监督目标
   - `thought` 作为辅助文本监督
   - 如果实现成本过高，第一版可先不训练 thought，只保留原始字段用于后续蒸馏

3. 过滤明显脏样本
   - bbox 占位符未替换
   - tool 调用失败
   - observation 与 action 不一致
   - 最终答案缺失

4. 先产出一个可训练的 SFT 数据集格式
   这里优先采用 verl 兼容的数据组织方式，但不强依赖官方样板；只要能被训练脚本稳定读取即可。

交付物建议：

- `src/train/RFT/build_sft_dataset.py`
- `src/train/RFT/data/controller_sft_train.jsonl`
- `src/train/RFT/data/controller_sft_val.jsonl`

## Phase 2：verl 训练入口与 policy 头改造

目标：让模型从“自由代码生成”切到“结构化动作决策”。

关键决定：

- 不直接复用 `ReactCodeAgent` 作为训练主体
- 训练时单独定义一个 `ControllerPolicy`
- 推理时再把 `action_type + action_args` 转成实际 tool call

实现重点：

1. 建立 action head
   - head 1：`action_type` 分类
   - head 2：参数生成/分类
   - 对 `Navigate` 可直接离散分类
   - 对 `ObjectLocation2D/3D`、`VisualQA`、`FinalAnswer` 的参数，先做模板化，减少自由生成空间

2. 建立 action renderer
   把结构化动作恢复为当前仓库可执行形式，例如：
   - `Navigate(move_forward)` -> `GoNextPointTool("move_forward")`
   - `FinalAnswer(C)` -> `final_answer("C")`

3. 保留 reference policy
   RL 阶段需要一个 SFT reference model，用于 KL 约束，防止策略漂移后工具乱调。

交付物建议：

- `src/train/RFT/controller_policy.py`
- `src/train/RFT/action_renderer.py`
- `src/train/RFT/train_controller_sft.py`
- `src/train/RFT/configs/controller_sft.yaml`

## Phase 3：Offline RL 数据与奖励构造

目标：先不依赖大量在线 simulator rollout，把 reward 设计跑通。

奖励函数按设计文档拆成 8 项，但实现顺序必须分层推进。

### 第一批必须先做的 reward

1. `r_ans`
   - 最终答案正确：`+1.0`
   - 错误：`0` 或 `-1.0`

2. `r_redund`
   - 同一工具同一参数重复调用：负奖励
   - 已有结果却再次做相同定位 / crop / vqa：负奖励

3. `r_invalid`
   - 参数解析失败
   - 动作与当前状态不匹配
   - tool 执行报错

4. `r_prem`
   - 证据不足就 `FinalAnswer`
   - 未发现关键对象就直接回答

### 第二批 reward

5. `r_find`
   - 首次发现 related object

6. `r_info`
   - 获得新属性、新 crop、新 VQA 结论

### 第三批 reward

7. `r_prog`
   - 向 planner 子目标或目标物体更接近
   - 若 geodesic 不稳定，先用 waypoint progress 近似

8. `r_hall`
   - 没有观测依据却给出属性/状态/答案
   - 第一版可以先用启发式 verifier，不必一上来训练单独 reward model

离线负样本构造按设计文档执行：

- 提前 FinalAnswer
- 删除关键工具调用
- 重复无效导航
- 错对象上做定位
- 打乱关键步骤顺序
- 用错误答案结束

交付物建议：

- `src/train/RFT/reward_fn.py`
- `src/train/RFT/build_offline_rl_dataset.py`
- `src/train/RFT/verifiers.py`
- `src/train/RFT/configs/controller_offline_rl.yaml`

## Phase 4：Online RL 环境封装到 verl

目标：把当前 Habitat 交互流程包成一个 verl 可 rollout 的环境。

这里的核心不是“把整个 agent 原封不动接进去”，而是把 RL 需要的最小接口明确化：

- `reset(sample)`：
  初始化问题、场景、planner、起点、memory
- `step(action)`：
  执行动作 -> 调工具 -> 更新环境 -> 返回 reward / next state / done / info

环境内部建议复用：

- `src/runs/eqa_modeling.py`
- `src/tools/*.py`
- `src/agents/tool_eqa_agent_mp.py` 中已有的 step 逻辑，但要拆掉对自由代码生成的依赖

在线 rollout 的 episode 流程：

1. 读入样本
2. planner 生成固定 `plan`
3. controller 输出结构化动作
4. action renderer 转成 tool 调用
5. executor 执行并收集 observation
6. reward 函数打分
7. 达到 `FinalAnswer` 或 budget 用尽则结束

交付物建议：

- `src/train/RFT/tool_eqa_env.py`
- `src/train/RFT/rollout_manager.py`
- `src/train/RFT/configs/controller_online_rl.yaml`

## Phase 5：verl 算法选择与训练顺序

设计文档推荐 PPO，我同意这个选择。第一版不追求算法花样，优先稳定。

训练顺序：

1. `SFT warm start`
   先得到可执行、可收敛的 controller policy

2. `Offline PPO / advantage-weighted style update`
   在静态轨迹和构造负样本上验证 reward 是否朝正确方向推动策略

3. `Online PPO`
   在 Habitat rollout 上继续优化 stopping、tool use、exploration

为什么先不用更复杂算法：

- 当前最大风险不在算法上，而在动作建模和 reward 正确性
- PPO 更容易和审稿叙事对齐
- verl 对 actor/reference/rollout 的工程支持更成熟，便于快速验证

训练时要加的稳定项：

- KL 到 SFT reference policy
- action mask，禁止明显非法动作
- 每类 action 的最小采样覆盖，避免 policy 迅速塌到 `FinalAnswer`
- curriculum：先短 budget，再长 budget

## 4. 与当前仓库代码的具体对接点

### 4.1 需要优先复用的模块

- `src/agents/tool_eqa_agent_mp.py`
  参考其 `step()` 的 prompt、图像、日志组织方式

- `src/tools/tool_box.py`
  直接作为动作空间来源

- `src/runs/eqa_modeling.py`
  复用 Habitat 初始化、路径、观测和场景管理

- `data/ToolTrajectory/trainval.json`
  作为 SFT 和 Offline RL 初始数据

### 4.2 需要避免直接复用的部分

- 直接训练 `thought + code` 自由生成
  这会让 RL 信号过稀、动作空间过大、工具调用难以约束

- 直接对 planner 做 RL
  当前阶段收益不确定，工程成本也更高

## 5. 里程碑

### Milestone 1：数据闭环

完成标准：

- 能从 `trainval.json` 解析出 step-level 结构化动作
- 能产出一份干净的 SFT 数据集
- 能统计各类动作分布与脏数据比例

### Milestone 2：SFT baseline

完成标准：

- 模型在离线验证集上能稳定预测正确 action type
- 推理时能把结构化动作还原成现有 tool 调用
- 在小样本上能跑通完整 episode

### Milestone 3：Offline RL baseline

完成标准：

- reward 计算正确
- 相比纯 SFT，重复工具调用与 premature answer 明显下降

### Milestone 4：Online RL baseline

完成标准：

- 在真实 Habitat rollout 中稳定训练
- 不出现大规模 collapse 到乱导航或秒答

### Milestone 5：最终评估

完成标准：

- 对比 SFT baseline，至少观察以下指标中的一部分稳定提升：
  - answer accuracy
  - trajectory success
  - path efficiency
  - tool redundancy
  - premature stopping rate

## 6. 评估指标

除最终答题准确率外，我会重点看以下 RL 相关指标：

- `Answer Accuracy`
- `Success@Episode`
- `Average Steps`
- `Tool Calls / Episode`
- `Redundant Tool Call Rate`
- `Premature FinalAnswer Rate`
- `Evidence Sufficiency Before Answer`
- `Navigation Progress`

如果只看最终答案正确率，很难判断 RL 到底是在优化探索，还是只是在碰运气。

## 7. 风险与对应处理

### 风险 1：动作空间仍然过大

处理：

- 第一版只保留最常用的 5 到 6 类动作
- 参数模板化，不直接开放自由文本参数

### 风险 2：reward 不稳定或相互冲突

处理：

- 按“先硬约束、后软增益”的顺序实现 reward
- 先确保 `invalid / redundant / premature` 三项有效，再加 `find / info / prog`

### 风险 3：online rollout 太慢

处理：

- 先做 Offline RL 验证 reward 方向
- online 阶段先跑小场景、小 budget、小 batch

### 风险 4：策略塌缩到早停

处理：

- 引入 evidence sufficiency 检查
- 加强 premature penalty
- 加 reference KL 和 action mask

## 8. 我建议的实际落地顺序

为了尽快拿到第一个可运行版本，我会按下面顺序推进：

1. 写 `react -> structured action` 解析器
2. 生成 step-level SFT 数据集
3. 训练一个只输出结构化动作的 controller SFT baseline
4. 实现最小 reward：`ans + invalid + redund + prem`
5. 做 Offline RL
6. 再接 Habitat online rollout
7. 最后补 `find / info / prog / hall`

## 9. 预期新增文件清单

建议在 `src/train/RFT/` 下新增：

- `verl_controller_rft_plan.md`
- `action_schema.py`
- `parse_react_trajectory.py`
- `evidence_memory.py`
- `build_sft_dataset.py`
- `controller_policy.py`
- `action_renderer.py`
- `train_controller_sft.py`
- `reward_fn.py`
- `verifiers.py`
- `build_offline_rl_dataset.py`
- `tool_eqa_env.py`
- `rollout_manager.py`
- `configs/controller_sft.yaml`
- `configs/controller_offline_rl.yaml`
- `configs/controller_online_rl.yaml`

## 10. 最终判断

这个 RFT 方案在你当前仓库里是可落地的，但前提是先把 controller 从“代码生成 agent”改造成“结构化动作策略”。只要这一步做对，verl 主要承担的是训练编排与 PPO 优化，不会成为主要难点。真正的难点是：

- 结构化动作建模是否合理
- reward 是否能代表“证据充分后再回答”
- online 环境封装是否足够稳定

因此，第一阶段我不会急着“先把 verl 跑起来”，而会先把数据接口、动作空间、奖励闭环做扎实。这样后面的 SFT、Offline RL 和 Online RL 才有机会稳定提升。
