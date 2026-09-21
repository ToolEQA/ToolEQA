"""Prepare complete official splits and score without training reward filters."""

import argparse
import json
from pathlib import Path

from src.train.RFT.dataset import build_record, iter_json_array
from src.evaluation.paper_metrics import compute_paper_metrics
from src.train.RFT.reward import answer_is_correct
from src.train.RFT.reward_fn import _extract_runtime_fields
from src.train.RFT.trajectory_log import attach_reward
from src.evaluation.resume_official import hardware_errors


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    sample, trace, answer, runtime = _extract_runtime_fields(extra_info)
    failures = hardware_errors(trace)
    if failures:
        raise RuntimeError("Hardware failure invalidates this evaluation; stop and requeue: " + failures[0])
    sample = dict(sample)
    sample.setdefault("answer", ground_truth)
    # Official recall uses the original annotations, without reward auditing.
    sample.pop("evidence_targets", None)
    judgment = None
    if sample.get("answer_setting") == "open":
        from src.evaluation.open_protocol import semantic_judgment
        judgment = semantic_judgment(str(sample["question"]), str(sample["answer"]), str(answer or ""))
    correct = judgment["score"] == 5 if judgment is not None else answer_is_correct(answer, sample)
    quality = judgment["score"] / 5.0 if judgment is not None else None
    attempts = [s for s in trace if s.get("action_type") in
                {"Navigate", "Location2D", "Location3D", "Crop", "VisualQA"}]
    successful = sum(s.get("ok") is not False for s in attempts)
    result = {
        "score": quality if quality is not None else float(correct), "acc": float(correct),
        "tool_attempt_count": len(attempts),
        "successful_tool_call_count": successful,
        "tool_call_accuracy": successful / len(attempts) if attempts else 0.0,
        "final_answer_called": float(answer is not None),
        "forced_final": float(bool(runtime.get("forced_final", False))),
        "truncated_normal_action": float(bool(runtime.get("truncated_normal_action", False))),
        **compute_paper_metrics(sample, trace, correct=correct, answer_quality=quality),
    }
    if judgment is not None:
        result.update(llm_match=quality, semantic_score=judgment["score"])
    attach_reward(runtime.get("trajectory_id"), runtime.get("global_step"), result,
                  path_override=runtime.get("trajectory_log_path"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = (root / "config/react-eqa.yaml").read_text()
    config = config.replace("scene_data_path: [./data/HM3D, ./data/OpenEQA/scenes]",
                            f"scene_data_path: [{args.scene_root}]")
    (args.output_dir / "env.yaml").write_text(config)
    report = {}
    for split in ("seen", "unseen"):
        source = root / "data/ToolTrajectory" / f"{split}_testset.json"
        rows = []
        missing = set()
        affected = 0
        for index, sample in enumerate(iter_json_array(source)):
            scene_dir = args.scene_root / sample["scene"]
            if not (scene_dir / f'{sample["scene"][6:]}.basis.glb').is_file() or not (
                    scene_dir / f'{sample["scene"][6:]}.basis.navmesh').is_file():
                missing.add(sample["scene"])
                affected += 1
            record = build_record(sample, index)
            record["extra_info"]["official_test_split"] = split
            rows.append(record)
        report[split] = {"samples": len(rows), "missing_scenes": sorted(missing),
                         "affected_samples": affected}
        target = args.output_dir / f"{split}.jsonl"
        with target.open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} samples, no filtering; {target}")
    (args.output_dir / "scene_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    for split, audit in report.items():
        print(f'{split}: {len(audit["missing_scenes"])} missing scenes, '
              f'{audit["affected_samples"]} affected samples')
    if any(audit["missing_scenes"] for audit in report.values()):
        raise SystemExit("Full evaluation blocked by missing scene assets; see scene_audit.json")


if __name__ == "__main__":
    main()
