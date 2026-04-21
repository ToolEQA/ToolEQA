import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _build_prompt(sample: Dict) -> str:
    proposals = sample.get("proposals") or []
    choice_block = ""
    if proposals:
        choice_lines = [f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(proposals)]
        choice_block = "\nChoices:\n" + "\n".join(choice_lines)

    plan = (sample.get("plan") or "").strip()
    return (
        "You are the ToolEQA controller.\n"
        "You must reason over the current scene, choose the next tool/action, and stop only when evidence is sufficient.\n\n"
        f"Question:\n{sample['question']}\n\n"
        f"Planner Plan:\n{plan}\n"
        f"{choice_block}\n\n"
        "Output the next action as JSON:\n"
        '{"action_type": "...", "args": {...}}'
    )


def _build_chat_prompt(sample: Dict) -> List[Dict[str, str]]:
    return [{"role": "user", "content": _build_prompt(sample)}]


def _convert_sample(sample: Dict) -> Dict:
    return {
        "data_source": "tooleqa",
        "prompt": _build_chat_prompt(sample),
        "ability": "embodied_question_answering",
        "reward_model": {"style": "function", "ground_truth": sample.get("answer", "")},
        "extra_info": {
            "sample_id": sample["sample_id"],
            "scene": sample["scene"],
            "question": sample["question"],
            "answer": sample.get("answer"),
            "plan": sample.get("plan", ""),
            "related_objects": sample.get("related_objects", []),
            "proposals": sample.get("proposals", []),
            "init_pos": sample.get("init_pos"),
            "init_rot": sample.get("init_rot"),
            "traj_length": sample.get("traj_length"),
            "agent_name": "tooleqa",
        },
    }


def iter_samples(source_path: str, limit: Optional[int] = None) -> Iterable[Dict]:
    data = json.load(open(source_path, "r", encoding="utf-8"))
    if limit is not None:
        data = data[:limit]
    for sample in data:
        yield _convert_sample(sample)


def build_verl_dataset(source_path: str, output_path: str, limit: Optional[int] = None) -> List[Dict]:
    records = list(iter_samples(source_path, limit=limit))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to ToolEQA source json.")
    parser.add_argument("--output", required=True, help="Path to verl-compatible jsonl output.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    args = parser.parse_args()

    records = build_verl_dataset(args.source, args.output, limit=args.limit)
    print(f"Saved {len(records)} samples to {args.output}")
