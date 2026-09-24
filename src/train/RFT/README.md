# Evidence-Grounded RFT for ToolEQA

This directory implements the RFT stage proposed after the EMNLP submission:
the paper's Planner → Controller → Spatial Memory → Executor architecture is
kept intact, while a Qwen3-VL-8B-Instruct controller is optimized with GRPO
using verified evidence-state changes.

The old implementation rewarded requested object names, every Crop/VQA call,
and exact action repetition heuristics. It was deleted because those signals
could be increased without solving the embodied task. The replacement never
rewards a tool call merely for occurring.

## Reward definition

For rollout `τ`, let `C=Φ(mT)` be verified task-aware evidence coverage. The
joint-stage objectives are:

```text
Revidence = 2.0 C + sufficiency costs + C × efficiency costs
Ranswer   = C × (+1 if correct else -1)
```

VERL uses GDPO to normalize these dimensions independently within each prompt
group. A correct unsupported guess therefore has the same answer objective as
a wrong unsupported guess: zero. If all rollouts have zero evidence, both
objectives are constant and the actor receives zero advantage instead of
learning cheap guessing. The evidence-acquisition stage sets the answer
dimension weight to zero; the joint stage enables both dimensions.

`Φ(m)` is task-aware evidence coverage:

| Question family | Required evidence per related object |
| --- | --- |
| size | grounding + verified 3D position + positive 3D size; all operands close a comparison fact |
| distance | grounding + verified 3D position; all operands close a pairwise-distance fact |
| counting | grounding + distinct verified 3D instance positions; the complete set closes a count fact |
| location-location, location-special | grounding + verified 3D position + a visual observation of semantic context |
| color, special attribute, status, relationship | grounding + an object-referencing VQA observation from the same view |
| unknown/new types | grounding, with no guessed tool sequence |

`ObjectLocation3D` measurements are accepted only in the Habitat world frame
and only when the returned center is within the configured tolerance of the
training annotation. Position and size are separate facts: distance/counting
need positions, while dimension comparison additionally needs positive sizes.
The online tool anchors each DetAny3D detection to the depth map captured with
the same Habitat RGB frame before transforming it into world coordinates. This
avoids treating DetAny3D's monocular absolute-depth estimate as simulator
ground truth; DetAny3D still supplies open-vocabulary grounding and metric box
dimensions, normalized to `[length, width, height]`.
The audit records verified measurements, pairwise distances, volumes, heights,
and deduplicated instance counts. These annotations are privileged reward-side
inputs and never enter the policy prompt.

Reward targets are materialized only in derived RFT manifests. Conversion
copies candidate `related_objects` into `evidence_targets`, then records
`reward_eligible` and a machine-readable `reward_audit`. It rejects duplicate
answer options, missing or coincident operands, object labels that cannot be
grounded in the question, attribute operands that occur only as contextual
objects (for example, `table` in `lamp on the table`), and semantic duplicate
tasks. Rejected rows go to `reward_quarantine.jsonl`; the original EQA-RT JSON
is never edited. Both preflight and reward computation fail closed if these
explicit audit fields are absent or invalid.

The missing-answer term defaults to `-2.0`, so omitting `final_answer` is not a
safe fallback. A forced final turn has its own per-turn budget and records
`forced_final` separately.

The defaults are in
[`configs/evidence_grpo.yaml`](./verl_adapter/configs/evidence_grpo.yaml).
All components are logged separately (`acc`, `evidence_coverage`, invalid,
duplicate, no-progress, path cost, and so on). `compute_reward` also returns a
full per-step audit for offline analysis.

Evidence is paid once through terminal sufficiency. Per-step deltas remain
diagnostic; because the state is monotonic, summing them would reproduce the
same terminal coverage. GDPO supplies trajectory-level multi-objective
advantages, not token-level process supervision.

## What is implemented

- `evidence.py`: deterministic task specification, evidence state, tool-result
  validation, task-specific 3D operands, duplicate/context handling, and trace
  replay.
- `reward.py` / `reward_fn.py`: auditable reward plus the flat numeric VERL API.
- `verl_adapter/agent_loop.py`: a Python controller loop using the original
  single-pass Thought–Code action protocol. Each turn generates one
  `Thought: ... Code: ... <end_action>` continuation, executes the Python Code,
  and carries only the resulting Observation and Spatial Memory into later
  prompts. Tool-free Python computation remains a valid action. Each
  trajectory stores its full trace for reward calculation and uniformly
  samples one exact turn prompt/completion for PPO, avoiding both
  rollout/training context mismatch and extra weight for longer trajectories.
  The rollout context is 16K, with at most 12K persistent prompt tokens and a
  768-token cap for each complete Thought–Code action.
- `trajectory_log.py`: one atomic JSON record per rollout, containing every
  local thought, Code block, Observation, tool event, Spatial Memory snapshot,
  termination state, and the exact reward audit attached by the reward worker.
- `dataset.py`: streaming conversion of the 777 MB source JSON, evidence-target
  auditing, quarantine output, and semantic-task deduplication; related objects,
  positions, and answers remain privileged reward metadata and are not inserted
  into the policy prompt.
- `preflight.py`: checkpoint, dataset, simulator assets, Python stack, and
  DetAny3D liveness checks.
- `tests/`: reward-hacking regression tests.

The persistent Spatial Memory is instance-aware for 3D detections. Repeated
detections near the same world coordinate update one entry, while distinct
same-category objects receive stable keys such as `bed` and `bed#2`. This is
required for counting and for reasoning over multiple objects of one category.
It also persists compact recent tool signatures and observations. A task
checklist and next-action recommendations are enabled only when a deployable
planner supplies non-privileged `policy_targets`; reward-side
`evidence_targets` never enter policy-visible memory. Exact duplicate
perception calls are rejected before tool execution; repeated navigation
directions remain valid when the current viewpoint changed.

Image-taking tools are bound to the active `GoNextPointTool` episode. A stale
`next_point_N.jpg` left by another GRPO sample is replaced with the current
registered view, while a DetAny3D no-detection result consistently returns
`([], [])` rather than changing the two-value API contract.

## Prepare data

The converter makes a deterministic, disjoint, question-type-stratified split:
25 audited examples for each of the nine types form the 225-example development
set, and all remaining eligible examples stay in training. The official Seen
and Unseen test sets remain untouched for final evaluation.

```bash
src/train/RFT/scripts/prepare_data.sh
src/train/RFT/scripts/prepare_balanced_manifests.sh
```

For a quick converter check:

```bash
src/train/RFT/scripts/prepare_data.sh --limit 100 --val-per-question-type 0 \
  --output-dir /tmp/tooleqa-rft-data
python -m unittest discover -s src/train/RFT/tests -v
python -m src.train.RFT.dry_run
```

The default outputs are `train_reward_eligible.jsonl` (11,374 examples),
`validation_reward_eligible.jsonl` (225 examples), and
`reward_quarantine.jsonl`. The balanced training curriculum contains 50
examples per type (450 total); periodic online validation contains five per
type (45 total).

## GPU layout and services

This machine currently exposes five CUDA-usable L40 GPUs. Because broken NVML
entries shift the final CUDA ordinal, the tested Ray mask is `0,1,2,4`: three
cards for FSDP training and one for asynchronous vLLM rollout. Physical GPU 3
is reserved for one DetAny3D worker. DetAny3D communicates on
shared-memory channel 0, so its physical GPU need not share an index with the
Ray process.

In terminal 1:

```bash
src/train/RFT/scripts/run_detany3d.sh
```

The script removes only stale IPC objects for its selected channel before
starting. Override `DETANY_PYTHON`, `DETANY_GPU`, or `TOOLEQA_TOOL_GPU_ID` when
using another installation/layout.

## Train

The default actor and rollout model is the untouched local
`Qwen3-VL-8B-Instruct` checkpoint at
`/mynvme0/models/Qwen/Qwen3-VL-8B-Instruct`. The rollout asks it to emit the
same single-pass Thought–Code action used by the repository's inference agent;
it does not use Qwen's `<think>` mode or a separate Code generation call.
`MODEL_PATH` remains optional so an ablation can explicitly select another
complete Hugging Face checkpoint; a standalone LoRA adapter is not accepted by
vLLM.

```bash
src/train/RFT/scripts/run_evidence_grpo.sh
```

Print the fully resolved Hydra configuration without launching workers:

```bash
DRY_RUN=1 SKIP_DETANY_CHECK=1 src/train/RFT/scripts/run_evidence_grpo.sh
```

Common overrides can be appended directly:

```bash
MODEL_PATH=/path/to/checkpoint src/train/RFT/scripts/run_evidence_grpo.sh \
  actor_rollout_ref.rollout.n=6 \
  data.train_batch_size=1 \
  reward.custom_reward_function.reward_kwargs.evidence_terminal=0.0
```

The last override gives the answer-only ablation. For the paper, report answer
accuracy, evidence coverage/completion, grounded/position/size/visual fact
counts, path length, invalid/repeated action rates, tool count, and Seen/Unseen
generalization—not only the optimized return.

All new run artifacts default to
`/mynvme0/ToolEQA_RFT/<experiment_name>`. The repository-side output path is
no longer used for new checkpoints. Full trajectory diagnostics are written to
the run's `trajectories/step_N/` directory.

Run deterministic fixed-set evaluation before and after a pilot with:

```bash
src/train/RFT/scripts/run_fixed_eval.sh base
src/train/RFT/scripts/merge_fsdp_checkpoint.sh \
  /mynvme0/ToolEQA_RFT/eqa-rt-rft-instruct-thought-code-stage1-balanced450/checkpoints/global_step_450
src/train/RFT/scripts/run_fixed_eval.sh checkpoint
```

The merge command converts the rank-local FSDP actor shards into a standard
Hugging Face checkpoint without changing the original optimizer-bearing
checkpoint. Checkpoint evaluation defaults to the merged stage-1 step-450 model
and starts a fresh validation-only job. Both modes use the same balanced,
audited 225-example validation file, one greedy trajectory per example, and separate output
directories. Summarize either directory with
`python -m src.train.RFT.summarize_rollouts <validation-dir>`.

Training is staged. First train evidence acquisition from the untouched model:

```bash
src/train/RFT/scripts/run_stage1_evidence.sh
```

After merging the selected stage-1 checkpoint, run the 100-step joint pilot
with gated answer reward and the largest useful rollout group for the current
three-rank/one-environment layout (`n=12`):

```bash
src/train/RFT/scripts/run_joint_pilot_100.sh
```

It evaluates before training and at steps 25/50/75/100. Validation records
include the paper metrics (success, Recall@5/10/15, EPath@5/10/15, trajectory
length), raw equation-(5) recall, evidence metrics, and protocol-level tool
call accuracy. Only the two newest full checkpoints are retained because each
optimizer-bearing checkpoint occupies about 50 GB.

Stage 1 traverses a 450-example balanced curriculum once with six rollouts per
prompt. Its first three 50-example blocks introduce visual grounding,
single-object 3D tasks, and pairwise 3D tasks; the remaining 300 examples mix
all nine types. It validates on a balanced 45-example subset every 50 steps and
disables answer advantage and early efficiency costs. Merge its final actor
before the joint stage because native FSDP resume is broken on this host:

```bash
src/train/RFT/scripts/merge_fsdp_checkpoint.sh \
  /mynvme0/ToolEQA_RFT/eqa-rt-rft-v6-evidence-stage1-balanced450/checkpoints/global_step_450
src/train/RFT/scripts/run_stage2_joint.sh
```

The 450-step joint stage gates answer learning by verified coverage and
restores small no-progress, tool, and path costs. Run the full 225-example
fixed evaluation at each stage boundary. A shorter 100-step diagnostic remains
available as:

```bash
src/train/RFT/scripts/run_pilot_100.sh
```

It uses six GRPO trajectories per prompt, validates every 50 steps, saves every
50 steps, and retains only the newest actor checkpoint. Each checkpoint is
about 50 GB because optimizer state is included. On this host, PyTorch 2.9
currently segfaults while restoring the rank-local FSDP DTensor state, so the
pilot deliberately starts with `resume_mode=disable`. Saved actor shards can be
merged for evaluation with `merge_fsdp_checkpoint.sh`; do not claim optimizer
resume support until the native FSDP restore smoke test passes.

## Required external assets

Online RFT needs the HM3D and OpenEQA scene directories configured by
`config/react-eqa.yaml`. They are intentionally not stored in Git. Preflight
fails rather than silently running a non-embodied/debug rollout when those
directories are absent.
# Shared evaluation modules

The answer judge now defaults to `http://127.0.0.1:18942` (frozen original
Qwen3-VL-8B-Instruct), configurable via `TOOLEQA_FROZEN_JUDGE_SERVICE`.
`TOOLEQA_FROZEN_SERVICE` remains the planner endpoint (port 18941, unchanged 7B).
Use a separate cache for the 8B judge; historical scores used the 7B judge and
are not retroactively changed. Do not mix judge models in a comparison.

As of the OpenEQA-aligned protocol, the judge uses the upstream `mmbench` or
`mmbench-extra` prompt as one user message, integer marks 1--5, temperature 0.2,
seed 1234 per request, and 32 output tokens. Normalized quality is
`(clip(mark, 1, 5) - 1) / 4`; percentage LLM-Match is 100 times its mean.
The upstream last-period answer preprocessing and None-prediction handling are
preserved. Empty strings and bare option letters are sent to the judge rather
than assigned a custom zero. The backend is local Qwen3-VL-8B, not official GPT-4;
PyTorch seeds do not reproduce OpenAI API randomness. New caches and judge
protocol IDs separate this from all historical 0--5 scores. Planner requests
retain their existing protocol. No old trajectories, rewards or tables are
rewritten; re-score saved final answers before comparing new-protocol results.
The reward helper retains legacy normalization by default for historical
replay; online scoring explicitly selects the OpenEQA scale.

Python evaluation implementations now live in `src/evaluation/`:
`paper_metrics.py`, `open_protocol.py`, `frozen_service.py`, `official_eval.py`,
`summarize_rollouts.py`, `resume_official.py`, and `select_open_checkpoint.py`.
The matching RFT modules remain compatibility aliases; reward training stays here.
Metric formulas, judge prompts, protocol IDs and cache keys are unchanged.

Preview full evaluation without launching jobs:

```bash
python -m src.evaluation.run_open_eval --run-root /path/to/run
python -m src.evaluation.summarize_rollouts /path/to/validation/150.jsonl
```

Add `--execute` to launch full evaluation only after training finishes and
checkpoint-selection audits pass. The Python entry refuses active-run locks and
existing output files. It uses the existing training launch/merge shell helpers;
no shell scripts are placed in `src/evaluation/`.
The existing selector's strict duplicate-trajectory audit is unchanged: padded
development trajectories still require a separate audit fix before automatic
selection can succeed. This migration does not certify cross-split isolation.
