"""Select a saved joint checkpoint using complete, audited development runs only."""
import argparse
import json
from pathlib import Path
from statistics import fmean

from src.evaluation.open_protocol import PROTOCOL_ID
from src.evaluation.resume_official import hardware_errors


def select(run_root: Path):
    dev = [json.loads(line)["extra_info"] for line in (run_root / "data-v2/dev.jsonl").read_text().splitlines()]
    expected = {str(row["sample_id"]) for row in dev}
    if len(expected) != 225:
        raise ValueError("Expected the complete fixed 225-task development split")
    stage = run_root / "stage2-joint"
    candidates = []
    for step in (150, 300, 450):
        checkpoint = stage / "checkpoints" / f"global_step_{step}"
        if not (checkpoint / "actor/fsdp_config.json").is_file():
            raise ValueError(f"Missing saved joint checkpoint: {checkpoint}")
        table = [json.loads(line) for line in (stage / "validation" / f"{step}.jsonl").read_text().splitlines()]
        if len(table) != 225 or any("llm_match" not in row for row in table):
            raise ValueError(f"Incomplete development metrics at {step}")
        scored = {}
        for path in (stage / "trajectories" / f"step_{step}").glob("*.json"):
            row = json.loads(path.read_text())
            if not row.get("validate"):
                continue
            info, reward = row["sample"], row.get("reward", {})
            sid = str(info["sample_id"])
            if sid in scored or sid not in expected or hardware_errors(row["tool_trace"]):
                raise ValueError(f"Invalid/duplicate dev trajectory: {path}")
            if info.get("answer_setting") != "open" or info.get("planner_source") != "question-only-frozen-v1":
                raise ValueError(f"Development protocol mismatch: {path}")
            if info.get("frozen_protocol_id") != PROTOCOL_ID or "semantic_judgment" not in reward:
                raise ValueError(f"Missing frozen judgment: {path}")
            scored[sid] = float(reward["answer_quality"])
        if set(scored) != expected:
            raise ValueError(f"Incomplete development IDs at {step}: {len(scored)}/225")
        mean = fmean(scored.values())
        if abs(mean - fmean(float(row["llm_match"]) for row in table)) > 1e-8:
            raise ValueError("Development table and trace scores disagree")
        candidates.append({"step": step, "checkpoint": str(checkpoint), "dev_llm_match": mean})
    best = max(candidates, key=lambda row: (row["dev_llm_match"], -row["step"]))
    return {"selection_rule": "highest full-dev normalized LLM-Match; ties earliest step",
            "dev_count": 225, "protocol_id": PROTOCOL_ID, "candidates": candidates, **best}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    result = select(args.run_root)
    (args.run_root / "checkpoint-selection.json").write_text(json.dumps(result, indent=2))
    print(result["checkpoint"])


if __name__ == "__main__":
    main()
