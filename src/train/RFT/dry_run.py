"""Dependency-light sanity check for evidence tracking and reward ordering."""

from __future__ import annotations

import json

from src.train.RFT.reward import compute_reward


SAMPLE = {
    "sample_id": "dry-run",
    "question": "Is the chair larger than the table?",
    "question_type": "attribute-size",
    "proposals": ["yes", "no"],
    "answer": "A",
    "traj_length": 2.0,
    "related_objects": [
        {"name": "chair", "id": 1, "pos": [0.0, 0.0, 0.0]},
        {"name": "table", "id": 2, "pos": [1.0, 0.0, 0.0]},
    ],
    "evidence_targets": [
        {"name": "chair", "id": 1, "pos": [0.0, 0.0, 0.0]},
        {"name": "table", "id": 2, "pos": [1.0, 0.0, 0.0]},
    ],
    "reward_eligible": True,
    "reward_audit": [],
}


def location(name: str, center: list[float], step: int) -> dict:
    return {
        "step": step,
        "action_type": "Location3D",
        "args": {"object": name, "image_path": f"view-{step}.jpg"},
        "result": [[center], [[1.0, 1.0, 1.0]]],
        "ok": True,
        "image_path_before": f"view-{step}.jpg",
        "path_length_after": float(step),
    }


def main() -> None:
    guessed = compute_reward(
        SAMPLE,
        [{"step": 0, "action_type": "FinalAnswer", "args": {"answer": "yes"}, "ok": True}],
    )
    grounded = compute_reward(
        SAMPLE,
        [
            location("chair", [0.0, 0.0, 0.0], 0),
            location("table", [1.0, 0.0, 0.0], 1),
            {"step": 2, "action_type": "FinalAnswer", "args": {"answer": "yes"}, "ok": True},
        ],
    )
    repeated = compute_reward(
        SAMPLE,
        [location("chair", [0.0, 0.0, 0.0], index) for index in range(6)],
    )
    assert grounded["score"] > guessed["score"] > repeated["score"]
    print(json.dumps({"grounded": grounded, "guessed": guessed, "repeated": repeated}, indent=2))


if __name__ == "__main__":
    main()
