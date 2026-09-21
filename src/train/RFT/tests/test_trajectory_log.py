from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.train.RFT.select_fixed_validation import select_records, select_stratified_records
from src.train.RFT.reward_fn import score
from src.train.RFT.trajectory_log import attach_reward, write_trajectory
from src.train.RFT.verl_adapter.agent_loop import ToolEQAEvidenceAgentLoop


class TrajectoryLogTest(unittest.TestCase):
    @staticmethod
    def _eligible_info(sample_id: str, question_type: str = "object", **values):
        return {
            "sample_id": sample_id,
            "question": f"Where is the chair in task {sample_id}?",
            "question_type": question_type,
            "related_objects": [{"name": "chair", "id": sample_id, "pos": [0, 0, 0]}],
            "evidence_targets": [{"name": "chair", "id": sample_id, "pos": [0, 0, 0]}],
            "reward_eligible": True,
            "reward_audit": [],
            **values,
        }

    def test_atomic_trace_and_reward_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"TOOLEQA_TRAJECTORY_DIR": directory}
        ):
            trajectory_id = write_trajectory(
                {
                    "trajectory_id": "trajectory-1",
                    "global_step": 7,
                    "turns": [{"thought": "look", "code": "print(1)", "observation": "1"}],
                }
            )
            self.assertEqual(trajectory_id, "trajectory-1")
            self.assertTrue(attach_reward(trajectory_id, 7, {"score": 0.5, "correct": True}))
            path = Path(directory) / "step_7/trajectory-1.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["turns"][0]["thought"], "look")
            self.assertEqual(payload["reward"]["score"], 0.5)
            self.assertFalse(any(path.parent.glob("*.tmp-*")))

    def test_thought_extraction_supports_tagged_and_code_prefixed_text(self) -> None:
        extract = ToolEQAEvidenceAgentLoop._extract_thought
        self.assertEqual(extract("<think>inspect cup</think>Code:\n```py\npass\n```"), "inspect cup")
        self.assertEqual(extract("inspect cup\nCode:\n```py\npass\n```"), "inspect cup")
        self.assertEqual(extract("Thought: inspect cup\nCode:\n```py\npass\n```"), "inspect cup")
        self.assertEqual(extract("</think>\n```text\nCode:\n```py\npass\n```"), "")

    def test_complete_code_at_token_boundary_is_not_discarded(self) -> None:
        complete = "Thought: inspect cup\nCode:\n```py\nprint(1)\n```<end_action>"
        incomplete = "Thought: inspect cup\nCode:\n```py\nprint(1)"
        check = ToolEQAEvidenceAgentLoop._response_is_truncated
        self.assertFalse(check(complete, token_count=128, token_budget=128))
        self.assertTrue(check(incomplete, token_count=128, token_budget=128))
        self.assertFalse(check(incomplete, token_count=127, token_budget=128))

    def test_reward_worker_can_attach_by_absolute_path_without_shared_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step_3/trajectory-remote.json"
            with patch.dict(os.environ, {"TOOLEQA_TRAJECTORY_DIR": directory}):
                write_trajectory({"trajectory_id": "trajectory-remote", "global_step": 3, "turns": []})
            with patch.dict(os.environ, {}, clear=True):
                result = score(
                    data_source="tooleqa_evidence",
                    solution_str="",
                    ground_truth="A",
                    extra_info={
                        "trajectory_id": "trajectory-remote",
                        "global_step": 3,
                        "trajectory_log_path": str(path),
                        "sample_info": {
                            "answer": "A",
                            "question_type": "object",
                            "reward_eligible": True,
                            "reward_audit": [],
                            "evidence_targets": [
                                {"name": "chair", "id": 1, "pos": [0, 0, 0]}
                            ],
                        },
                        "tooleqa_trace": [
                            {
                                "step": 0,
                                "action_type": "FinalAnswer",
                                "args": {"answer": "A"},
                                "result": "A",
                                "ok": True,
                            }
                        ],
                        "tooleqa_final_answer": "A",
                    },
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["reward"]["score"], result["score"])
            self.assertTrue(payload["reward"]["correct"])

    def test_fixed_validation_selection_is_order_independent(self) -> None:
        records = [
            {"extra_info": self._eligible_info(sample_id), "prompt": []}
            for sample_id in ("a", "b", "c", "d")
        ]
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            first.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            second.write_text(
                "\n".join(json.dumps(record) for record in reversed(records)) + "\n", encoding="utf-8"
            )
            selected_first = select_records(first, count=2, seed=2026)
            selected_second = select_records(second, count=2, seed=2026)
            ids_first = {record["extra_info"]["sample_id"] for record in selected_first}
            ids_second = {record["extra_info"]["sample_id"] for record in selected_second}
            self.assertEqual(ids_first, ids_second)

    def test_stratified_validation_selects_each_question_type(self) -> None:
        records = [
            {"extra_info": self._eligible_info("a1", "size", traj_length=3.0)},
            {"extra_info": self._eligible_info("a2", "size", traj_length=1.0)},
            {"extra_info": self._eligible_info("b1", "distance", traj_length=4.0)},
            {"extra_info": self._eligible_info("b2", "distance", traj_length=2.0)},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "validation.jsonl"
            source.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            selected = select_stratified_records(
                source, "question_type", per_group=1, seed=7, rank_field="traj_length"
            )
        self.assertEqual(
            {item["extra_info"]["question_type"] for item in selected},
            {"size", "distance"},
        )
        self.assertEqual(
            {item["extra_info"]["sample_id"] for item in selected},
            {"a2", "b2"},
        )

    def test_stratified_validation_can_filter_before_ranking(self) -> None:
        records = [
            {"extra_info": self._eligible_info("bad", "size", traj_length=1)},
            {"extra_info": self._eligible_info("good", "size", traj_length=2)},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "validation.jsonl"
            source.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            selected = select_stratified_records(
                source,
                "question_type",
                per_group=1,
                seed=7,
                rank_field="traj_length",
                predicate=lambda record: record["extra_info"]["sample_id"] == "good",
            )
        self.assertEqual(selected[0]["extra_info"]["sample_id"], "good")


if __name__ == "__main__":
    unittest.main()
