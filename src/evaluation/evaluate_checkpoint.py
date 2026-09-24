"""Run complete Seen/Unseen tests for an explicitly selected checkpoint."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

from src.evaluation.run_open_eval import REPO_ROOT, evaluation_env
from src.evaluation.resume_official import hardware_errors
from src.evaluation.summarize_rollouts import summarize


def audit_results(output: Path, manifest: Path):
    expected = {json.loads(line)["extra_info"]["sample_id"]
                for line in manifest.read_text().splitlines() if line.strip()}
    table = output / "validation/0.jsonl"
    rows = [json.loads(line) for line in table.read_text().splitlines() if line.strip()]
    traces = []
    for path in (output / "trajectories/step_0").glob("*.json"):
        trace = json.loads(path.read_text())
        if trace.get("validate"):
            traces.append((path, trace))
    accepted = set()
    retained = []
    for row in rows:
        matches = [(p, t) for p, t in traces if t.get("initial_image")
                   and t["initial_image"] in row["input"]]
        if len(matches) != 1:
            raise ValueError("Cannot uniquely resolve retained validation trajectory")
        path, trace = matches[0]
        sid = trace["sample"]["sample_id"]
        if sid in accepted or sid not in expected or hardware_errors(trace["tool_trace"]):
            raise ValueError(f"Invalid retained test trajectory: {path}")
        for key in ("semantic_score", "evidence_coverage", "evidence_complete",
                    "recall_at_5", "recall_at_10", "recall_at_15",
                    "epath_at_5", "epath_at_10", "epath_at_15", "trajectory_length"):
            if key not in row or abs(row[key] - trace["reward"][key]) > 1e-9:
                raise ValueError(f"Missing or inconsistent metric {key}: {path}")
        accepted.add(sid)
        retained.append({"sample_id": sid, "trajectory_path": str(path)})
    if accepted != expected or len(rows) != len(expected):
        raise ValueError(f"Incomplete test: {len(accepted)}/{len(expected)}")
    result = summarize(table)
    result.update(score_distribution=dict(Counter(r["semantic_score"] for r in rows)),
                  raw_score_scale="1-5", identity_audit="passed",
                  retained_trajectories=retained)
    return result


def completed_records(previous: Path, split: str, manifest: Path):
    expected = {r["extra_info"]["sample_id"]: r["extra_info"] for r in
                (json.loads(line) for line in manifest.read_text().splitlines() if line.strip())}
    grouped = {}
    for path in (previous / split / "trajectories").glob("step_*/*.json"):
        trace = json.loads(path.read_text())
        reward = trace.get("reward") or {}
        if not trace.get("validate") or "semantic_score" not in reward:
            continue
        sid = trace["sample"]["sample_id"]
        if sid not in expected:
            raise ValueError(f"Unexpected sample in recovery: {path}")
        if hardware_errors(trace["tool_trace"]):
            continue
        for key in ("question", "answer", "scene"):
            if trace["sample"][key] != expected[sid][key]:
                raise ValueError(f"Recovery sample mismatch: {path}: {key}")
        for key in ("semantic_score", "evidence_coverage", "evidence_complete", "trajectory_length"):
            if key not in reward:
                raise ValueError(f"Incomplete metrics in recovery: {path}")
        grouped.setdefault(sid, []).append((path, trace))
    tables = [json.loads(line) for table in (previous / split / "validation").glob("*.jsonl")
              for line in table.read_text().splitlines() if line.strip()]
    result = {}
    for sid, candidates in grouped.items():
        if len(candidates) > 1:
            candidates = [(p, t) for p, t in candidates if t.get("initial_image") and
                          any(t["initial_image"] in row.get("input", "") for row in tables)]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous padded recovery trajectories: {sid}")
        path, trace = candidates[0]
        result[sid] = {**trace["reward"], "sample_id": sid, "trajectory_path": str(path)}
    # Include earlier retained attempts when resuming a resumed run.
    inherited = previous / f"{split}-retained.jsonl"
    if inherited.is_file():
        for line in inherited.read_text().splitlines():
            row = json.loads(line)
            if row["sample_id"] in result or row["sample_id"] not in expected:
                raise ValueError("Duplicate/unexpected inherited retained sample")
            result[row["sample_id"]] = row
    return result


def execute(run_root: Path, checkpoint: Path, output: Path, merged_model: Path | None = None,
            resume_from: Path | None = None):
    from src.evaluation.openeqa_protocol import JUDGE_PROTOCOL_ID
    if not (checkpoint / "actor/fsdp_config.json").is_file():
        raise FileNotFoundError(checkpoint)
    with (run_root / "pipeline.lock").open("a") as training_lock, \
            (run_root / "evaluation.lock").open("a") as evaluation_lock:
        for lock in (training_lock, evaluation_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)

        def status(state, **details):
            value = {"state": state, "pid": os.getpid(),
                     "updated_at": datetime.now(timezone.utc).isoformat(), **details}
            temporary = output / "status.tmp"
            temporary.write_text(json.dumps(value, indent=2))
            temporary.replace(output / "status.json")
            print(json.dumps(value), flush=True)

        try:
            status("preflight")
            env = evaluation_env(run_root)
            # Physical 0: judge; 6: detector/planner. Physical 1/2: model
            # shards plus independent environments; physical 3: vLLM policy.
            env.update(RAY_VISIBLE_GPUS="1,2,3", TRAIN_GPUS="2", ROLLOUT_GPUS="1",
                       TOOLEQA_AGENT_GPU_ID="0", TOOLEQA_AGENT_GPU_MAP="0,1",
                       RFT_OUTPUT_ROOT=str(output),
                       TOOLEQA_PLANNER_REPAIR_LOG=str(output / "planner-repairs.jsonl"),
                       TOOLEQA_FROZEN_JUDGE_SERVICE="http://127.0.0.1:18942")
            with urllib.request.urlopen(env["TOOLEQA_FROZEN_JUDGE_SERVICE"], timeout=30) as response:
                judge = json.load(response)
            if judge["protocol_id"] != JUDGE_PROTOCOL_ID:
                raise ValueError("Wrong judge protocol")
            manifests = {}
            for split in ("seen", "unseen"):
                manifest = run_root / f"data-v2/{split}.jsonl"
                rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
                source = REPO_ROOT / f"data/ToolTrajectory/{split}_testset.json"
                source_rows = json.loads(source.read_text())
                ids = [r["extra_info"]["sample_id"] for r in rows]
                if len(set(ids)) != len(ids) or set(ids) != {r["sample_id"] for r in source_rows}:
                    raise ValueError(f"Incomplete/duplicated {split} manifest")
                for row in rows:
                    sample = row["extra_info"]
                    if sample.get("answer_setting") != "open":
                        raise ValueError("Not an open-vocabulary manifest")
                    scene = sample["scene"]
                    for suffix in ("basis.glb", "basis.navmesh"):
                        asset = Path(env["TOOLEQA_SCENE_ROOT"]) / scene / f"{scene[6:]}.{suffix}"
                        if not asset.is_file():
                            raise FileNotFoundError(asset)
                manifests[split] = {"path": str(manifest), "count": len(rows),
                                    "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
            metadata = {"checkpoint": str(checkpoint), "selection": "explicit latest step450",
                        "judge": judge, "splits": manifests, "include_counting": True,
                        "evidence_position_tolerance_m": 2.0,
                        "evidence_targets": "original related_objects, no eligibility filtering",
                        "gpu_allocation": {"0": "judge", "1,2": "shards and two environments",
                                           "3": "policy inference", "6": "detector and planner"}}
            metadata["planner_error_handling"] = "strip-preamble-or-generic-question-only-fallback-v1"
            if resume_from:
                prior = json.loads((resume_from / "manifest.json").read_text())
                if (prior["checkpoint"] != str(checkpoint) or prior["judge"] != judge
                        or prior["splits"] != manifests):
                    raise ValueError("Recovery checkpoint/judge/test manifest mismatch")
                metadata["resume_from"] = str(resume_from)
            (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
            merged = merged_model or output / "actor_huggingface"
            if merged_model is None:
                status("merging", checkpoint=str(checkpoint))
                with (output / "merge.log").open("x") as log:
                    subprocess.run(["bash", "src/train/RFT/scripts/merge_fsdp_checkpoint.sh",
                                    str(checkpoint), str(merged)], cwd=REPO_ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            else:
                original = json.loads((merged.parent / "manifest.json").read_text())
                if original["checkpoint"] != str(checkpoint):
                    raise ValueError("Merged model checkpoint provenance mismatch")
                index = json.loads((merged / "model.safetensors.index.json").read_text())
                for shard in set(index["weight_map"].values()):
                    if not (merged / shard).is_file():
                        raise FileNotFoundError(merged / shard)
            metadata["merged_model"] = str(merged)
            (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
            env.update(CHECKPOINT_PATH=str(checkpoint), MERGED_MODEL_PATH=str(merged))
            for split in ("seen", "unseen"):
                full_manifest = Path(manifests[split]["path"])
                retained = completed_records(resume_from, split, full_manifest) if resume_from else {}
                (output / f"{split}-retained.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in retained.values()))
                remaining = [json.loads(line) for line in full_manifest.read_text().splitlines()
                             if line.strip() and json.loads(line)["extra_info"]["sample_id"] not in retained]
                residual_manifest = output / f"{split}-remaining.jsonl"
                residual_manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in remaining))
                status("evaluating", split=split, total=manifests[split]["count"],
                       retained=len(retained), remaining=len(remaining))
                env.update(VAL_FILE=str(residual_manifest), EXPERIMENT_NAME=split)
                if remaining:
                    with (output / f"{split}.log").open("x") as log:
                        subprocess.run([
                        "bash", "src/train/RFT/scripts/run_fixed_eval.sh", "checkpoint",
                        f"reward.custom_reward_function.path={REPO_ROOT}/src/evaluation/official_eval.py",
                        "actor_rollout_ref.rollout.agent.num_workers=2",
                        "actor_rollout_ref.rollout.gpu_memory_utilization=0.72",
                        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1536",
                        "actor_rollout_ref.actor.use_kl_loss=false",
                        "data.val_batch_size=8",
                        ], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    result = audit_results(output / split, residual_manifest)
                    for item in result["retained_trajectories"]:
                        trace = json.loads(Path(item["trajectory_path"]).read_text())
                        retained[item["sample_id"]] = {**trace["reward"], **item}
                expected = {json.loads(line)["extra_info"]["sample_id"]
                            for line in full_manifest.read_text().splitlines() if line.strip()}
                if set(retained) != expected:
                    raise ValueError("Final recovery completeness mismatch")
                combined = output / f"{split}-all-scores.jsonl"
                combined.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in retained.values()))
                result = summarize(combined)
                result.update(score_distribution=dict(Counter(r["semantic_score"] for r in retained.values())),
                              raw_score_scale="1-5", identity_audit="passed")
                (output / f"{split}-summary.json").write_text(json.dumps(result, indent=2))
            status("completed")
        except BaseException as exc:
            status("failed", error=repr(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged-model", type=Path)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    execute(args.run_root.resolve(), args.checkpoint.resolve(), args.output_dir.resolve(),
            args.merged_model.resolve() if args.merged_model else None,
            args.resume_from.resolve() if args.resume_from else None)


if __name__ == "__main__":
    main()
