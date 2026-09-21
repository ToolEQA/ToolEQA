"""Resume official evaluation by sample identity, excluding hardware failures."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re


HARDWARE_ERROR = re.compile(
    r"CUDA error|CUDA out of memory|No CUDA GPUs|device.*(?:lost|fallen)|"
    r"DetAny.*(?:timed out|timeout)|共享内存.*超时", re.I)


def hardware_errors(trace):
    return [str(step.get("error", "")) for step in trace
            if HARDWARE_ERROR.search(str(step.get("error", "")))]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def source_rows(root, split):
    with (root / "data-mounted" / f"{split}.jsonl").open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare(root):
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    report = {"created_at": stamp, "splits": {}}
    for split in ("seen", "unseen"):
        rows = source_rows(root, split)
        expected = {row["extra_info"]["sample_id"] for row in rows}
        if len(expected) != len(rows):
            raise ValueError(f"Duplicate IDs in {split} manifest")
        accepted = set()
        quarantine = []
        for file in sorted((root / split / "trajectories").glob("step_*/*.json")):
            record = json.loads(file.read_text())
            sample_id = record["sample"]["sample_id"]
            if sample_id not in expected:
                raise ValueError(f"Unexpected sample ID in {file}")
            errors = hardware_errors(record.get("tool_trace", []))
            if errors or "acc" not in (record.get("reward") or {}):
                destination = root / "hardware_interrupted" / stamp / split / file.parent.name / file.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                file.rename(destination)
                quarantine.append({"sample_id": sample_id, "path": str(destination),
                                   "reason": errors or ["not fully scored"]})
                continue
            if sample_id in accepted:
                raise ValueError(f"Duplicate completed ID: {sample_id}")
            accepted.add(sample_id)
        remaining = [row for row in rows if row["extra_info"]["sample_id"] not in accepted]
        target = root / "data-resume" / f"{split}.jsonl"
        target.parent.mkdir(exist_ok=True)
        temporary = target.with_suffix(".tmp")
        with temporary.open("w") as handle:
            for row in remaining:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(target)
        report["splits"][split] = {"total": len(rows), "retained": len(accepted),
                                    "remaining": len(remaining), "archived": quarantine}
        print(f"{split}: retained={len(accepted)}, remaining={len(remaining)}, archived={len(quarantine)}")
    write_json(root / "recovery" / f"prepare-{stamp}.json", report)
    write_json(root / "recovery" / "latest.json", report)


def consolidate(root, split):
    from src.evaluation.summarize_rollouts import summarize
    expected = [row["extra_info"]["sample_id"] for row in source_rows(root, split)]
    records = {}
    for file in (root / split / "trajectories").glob("step_*/*.json"):
        record = json.loads(file.read_text())
        sample_id = record["sample"]["sample_id"]
        reward = record.get("reward") or {}
        if hardware_errors(record.get("tool_trace", [])) or "acc" not in reward:
            raise ValueError(f"Unusable trajectory: {file}")
        if sample_id in records:
            raise ValueError(f"Duplicate trajectory ID: {sample_id}")
        records[sample_id] = {**reward, "sample_id": sample_id, "step": 0,
                              "trajectory_id": record["trajectory_id"],
                              "trajectory_path": str(file)}
    if set(records) != set(expected):
        raise ValueError(f"{split} completeness mismatch: {len(records)}/{len(expected)}")
    target = root / split / "validation" / "0.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    for old in target.parent.glob("*.jsonl"):
        archive = root / split / "validation_attempts" / datetime.now().strftime("%Y%m%dT%H%M%S%f")
        archive.mkdir(parents=True)
        old.rename(archive / old.name)
    temporary = target.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for sample_id in expected:
            handle.write(json.dumps(records[sample_id], ensure_ascii=False) + "\n")
    temporary.replace(target)
    write_json(root / f"{split}-summary.json", [summarize(target.parent)])
    print(f"{split}: consolidated {len(records)} unique scored trajectories")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "consolidate"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split", choices=("seen", "unseen"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.run_root)
    else:
        if not args.split:
            parser.error("consolidate requires --split")
        consolidate(args.run_root, args.split)


if __name__ == "__main__":
    main()
