"""Prepare separate open-vocabulary records without changing the closed datasets."""
import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from src.train.RFT.dataset import build_record, iter_json_array
from src.train.RFT.open_protocol import PROTOCOL_ID


def convert(record, *, training=False):
    result = copy.deepcopy(record)
    info = result["extra_info"]
    label, options = str(info["answer"]).strip(), info.pop("proposals")
    if label not in "ABCD" or len(label) != 1 or ord(label) - 65 >= len(options):
        raise ValueError(f"Invalid original label: {info['sample_id']}")
    reference = str(options[ord(label) - 65]).strip()
    # These depend on withheld alternatives and require explicit reviewed rewrites.
    if re.search(r"\b(all of|none of|above (options|answers)|option [a-d]|both [a-d] and [a-d]|former|latter|first one|second one)\b", reference, re.I):
        raise ValueError(f"Reference needs a reviewed rewrite: {info['sample_id']}: {reference!r}")
    if not reference:
        raise ValueError("Empty reference")
    info["answer"] = reference
    info["answer_setting"] = "open"
    info["reference_source"] = "correct-option-text"
    info["frozen_protocol_id"] = PROTOCOL_ID
    info["planner_source"] = "stored-training-plan" if training else "question-only-frozen-v1"
    if not training:
        info.pop("plan", None)
    prompt = info["question"]
    if training and info.get("plan"):
        prompt += "\n\nPlanner guidance:\n" + info["plan"]
    result["prompt"] = [{"role": "user", "content": prompt}]
    result["reward_model"] = {"style": "rule", "ground_truth": reference}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    inputs = {
        "train": root / "src/train/RFT/data/train_staged_reward_eligible_450.jsonl",
        "dev": root / "src/train/RFT/data/validation_reward_eligible.jsonl",
        "seen": root / "data/ToolTrajectory/seen_testset.json",
        "unseen": root / "data/ToolTrajectory/unseen_testset.json",
    }
    datasets, manifest = {}, {"protocol_id": PROTOCOL_ID, "splits": {}}
    for split, path in inputs.items():
        if split in {"train", "dev"}:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            rows = [build_record(s, i) for i, s in enumerate(iter_json_array(path))]
            for row in rows:
                row["extra_info"]["official_test_split"] = split
        rows = [convert(row, training=split == "train") for row in rows]
        datasets[split] = rows
        manifest["splits"][split] = {"count": len(rows), "source": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "question_types": dict(Counter(r["extra_info"]["question_type"] for r in rows))}
    assert [len(datasets[k]) for k in inputs] == [450, 225, 845, 1069]
    train_keys = {(r["extra_info"]["scene"], r["extra_info"]["question"]) for r in datasets["train"]}
    dev_keys = {(r["extra_info"]["scene"], r["extra_info"]["question"]) for r in datasets["dev"]}
    if train_keys & dev_keys:
        raise ValueError("Train/development overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in datasets.items():
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        with (args.output_dir / f"{split}.jsonl").open("x") as handle:
            handle.write(payload)
        manifest["splits"][split]["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
