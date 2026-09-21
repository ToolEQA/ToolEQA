from __future__ import annotations

import unittest

from src.train.RFT.reward import answer_is_correct, compute_reward
from src.train.RFT.reward_fn import compute_score


SAMPLE = {
    "question": "Is the chair larger than the table?",
    "question_type": "attribute-size",
    "proposals": ["yes", "no"],
    "answer": "A",
    "traj_length": 2.0,
    "related_objects": [
        {"name": "chair", "id": 1, "pos": [0, 0, 0]},
        {"name": "table", "id": 2, "pos": [1, 0, 0]},
    ],
    "evidence_targets": [
        {"name": "chair", "id": 1, "pos": [0, 0, 0]},
        {"name": "table", "id": 2, "pos": [1, 0, 0]},
    ],
    "reward_eligible": True,
    "reward_audit": [],
}


def location(name: str, center: list[float], step: int, image: str | None = None) -> dict:
    return {
        "step": step,
        "action_type": "Location3D",
        "args": {"object": name, "image_path": image or f"{step}.jpg"},
        "result": [[center], [[1, 1, 1]]],
        "ok": True,
        "image_path_before": image or f"{step}.jpg",
        "path_length_after": float(step + 1),
    }


def final(answer: str, step: int = 99) -> dict:
    return {"step": step, "action_type": "FinalAnswer", "args": {"answer": answer}, "ok": True}


class RewardTest(unittest.TestCase):
    def test_option_letter_and_text_match(self) -> None:
        self.assertTrue(answer_is_correct("A", SAMPLE))
        self.assertTrue(answer_is_correct("The final answer is yes.", SAMPLE))
        self.assertTrue(answer_is_correct("Yes, the chair is larger.", SAMPLE))
        self.assertFalse(answer_is_correct("no", SAMPLE))

    def test_grounded_correct_beats_correct_guess(self) -> None:
        guess = compute_reward(SAMPLE, [final("yes")])
        grounded = compute_reward(
            SAMPLE,
            [location("chair", [0, 0, 0], 0), location("table", [1, 0, 0], 1), final("yes")],
        )
        self.assertGreater(grounded["score"], guess["score"])
        self.assertEqual(guess["components"]["unsupported_final"], -0.3)

    def test_answer_reward_is_gated_by_evidence(self) -> None:
        correct_guess = compute_reward(SAMPLE, [final("yes")])
        wrong_guess = compute_reward(SAMPLE, [final("no")])
        self.assertEqual(correct_guess["answer_objective"], 0.0)
        self.assertEqual(wrong_guess["answer_objective"], 0.0)

        evidence = [
            location("chair", [0, 0, 0], 0),
            location("table", [1, 0, 0], 1),
        ]
        correct = compute_reward(SAMPLE, evidence + [final("yes")])
        wrong = compute_reward(SAMPLE, evidence + [final("no")])
        self.assertEqual(correct["answer_objective"], 1.0)
        self.assertEqual(wrong["answer_objective"], -1.0)

    def test_evidence_phase_disables_answer_objective(self) -> None:
        trace = [location("chair", [0, 0, 0], 0), location("table", [1, 0, 0], 1), final("yes")]
        result = compute_reward(SAMPLE, trace, weights={"reward_phase": "evidence"})
        self.assertEqual(result["answer_objective"], 0.0)
        self.assertEqual(result["reward_phase"], "evidence")

    def test_wrong_final_beats_omitting_the_answer(self) -> None:
        missing = compute_reward(SAMPLE, [])
        wrong = compute_reward(SAMPLE, [final("no")])
        self.assertGreater(wrong["score"], missing["score"])
        self.assertEqual(missing["components"]["answer"], -2.0)

    def test_failed_and_repeated_calls_do_not_hack_reward(self) -> None:
        failed = {
            "action_type": "Location3D",
            "args": {"object": "chair"},
            "result": None,
            "ok": False,
            "error": "not found",
            "image_path_before": "same.jpg",
        }
        result = compute_reward(SAMPLE, [failed] * 8)
        self.assertLess(result["score"], 0)
        self.assertEqual(result["coverage"], 0)
        self.assertEqual(result["evidence"]["duplicate_count"], 7)

    def test_empty_or_noop_controller_turn_is_worse_than_no_attempt(self) -> None:
        no_attempt = compute_reward(SAMPLE, [])
        invalid_noop = compute_reward(
            SAMPLE,
            [
                {
                    "action_type": "InvalidCode",
                    "args": {},
                    "result": None,
                    "ok": False,
                    "error": "Controller code did not call a tool",
                }
            ],
        )
        self.assertLess(invalid_noop["score"], no_attempt["score"])
        self.assertEqual(invalid_noop["evidence"]["invalid_count"], 1)

    def test_valid_compute_is_neither_tool_cost_nor_new_evidence(self) -> None:
        compute = {
            "action_type": "Compute",
            "args": {"code": "print(2 + 2)", "input_signature": "abc"},
            "result": "4",
            "ok": True,
        }
        result = compute_reward(SAMPLE, [compute])
        self.assertEqual(result["coverage"], 0.0)
        self.assertEqual(result["evidence"]["tool_count"], 0)
        self.assertEqual(result["evidence"]["invalid_count"], 0)
        self.assertEqual(result["step_rewards"][0]["reward"], 0.0)

    def test_repeated_vqa_only_adds_evidence_once(self) -> None:
        sample = {
            "question": "What color is the chair?",
            "question_type": "attribute-color",
            "answer": "blue",
            "related_objects": [{"name": "chair", "id": 1, "pos": [0, 0, 0]}],
            "evidence_targets": [{"name": "chair", "id": 1, "pos": [0, 0, 0]}],
            "reward_eligible": True,
            "reward_audit": [],
        }
        grounding = {
            "action_type": "Location2D",
            "args": {"object": "chair"},
            "result": {"bboxes_2d": [[0, 0, 10, 10]]},
            "ok": True,
            "image_path_before": "same.jpg",
        }
        vqa = {
            "action_type": "VisualQA",
            "args": {"question": "What color is the chair?"},
            "result": "blue",
            "ok": True,
            "image_path_before": "same.jpg",
        }
        result = compute_reward(sample, [grounding] + [vqa] * 8)
        deltas = [step["delta"] for step in result["step_rewards"]]
        self.assertAlmostEqual(deltas[0], 1 / 3)
        self.assertAlmostEqual(deltas[1], 2 / 3)
        self.assertEqual(deltas[2:], [0.0] * 7)
        self.assertLess(result["score"], 0.0)  # no final answer plus repeat costs

    def test_terminal_evidence_is_not_double_counted_as_progress_reward(self) -> None:
        trace = [location("chair", [0, 0, 0], 0), location("table", [1, 0, 0], 1), final("yes")]
        result = compute_reward(SAMPLE, trace)
        self.assertNotIn("evidence_progress", result["components"])
        self.assertNotIn("evidence_progress", result["weights"])
        self.assertEqual(result["components"]["evidence_terminal"], 2.0)

    def test_path_excess_is_penalized(self) -> None:
        efficient = compute_reward(
            SAMPLE,
            [location("chair", [0, 0, 0], 0), location("table", [1, 0, 0], 1), final("yes")],
        )
        long_path_trace = [location("chair", [0, 0, 0], 0), location("table", [1, 0, 0], 1)]
        long_path_trace[-1]["path_length_after"] = 6.0
        long_path = compute_reward(SAMPLE, long_path_trace + [final("yes")])
        self.assertLess(long_path["score"], efficient["score"])

    def test_verl_interface_is_flat_and_numeric(self) -> None:
        trace = [location("chair", [0, 0, 0], 0), location("table", [1, 0, 0], 1), final("yes")]
        metrics = compute_score(
            data_source="tooleqa_evidence",
            solution_str="ignored",
            ground_truth="A",
            extra_info={"sample_info": SAMPLE, "tooleqa_trace": trace},
        )
        self.assertTrue(all(isinstance(value, (float, int)) for value in metrics.values()))
        self.assertEqual(metrics["acc"], 1.0)
        self.assertIn("evidence_objective", metrics)
        self.assertIn("answer_objective", metrics)

    def test_tool_call_accuracy_is_execution_diagnostic(self) -> None:
        trace = [
            {
                "action_type": "Navigate",
                "ok": True,
                "duplicate_rejected": False,
            },
            {
                "action_type": "InvalidCode",
                "ok": False,
            },
            final("yes"),
        ]
        result = compute_reward(SAMPLE, trace)
        self.assertEqual(result["tool_attempt_count"], 2)
        self.assertEqual(result["successful_tool_call_count"], 1)
        self.assertEqual(result["tool_call_accuracy"], 0.5)

    def test_reward_fails_closed_for_quarantined_manifest(self) -> None:
        sample = {
            **SAMPLE,
            "reward_eligible": False,
            "reward_audit": ["duplicate-choices"],
            "evidence_targets": SAMPLE["related_objects"],
        }
        with self.assertRaisesRegex(ValueError, "duplicate-choices"):
            compute_reward(sample, [final("yes")])

    def test_reward_fails_closed_for_unaudited_sample(self) -> None:
        sample = {
            key: value
            for key, value in SAMPLE.items()
            if key not in {"reward_eligible", "reward_audit", "evidence_targets"}
        }
        with self.assertRaisesRegex(ValueError, "unaudited sample"):
            compute_reward(sample, [final("yes")])

    def test_reward_fails_closed_if_targets_are_tampered_after_audit(self) -> None:
        sample = {
            **SAMPLE,
            "evidence_targets": [
                {"name": "lamp", "id": 9, "pos": [4, 0, 0]},
            ],
        }
        with self.assertRaisesRegex(ValueError, "do not match"):
            compute_reward(sample, [final("yes")])


if __name__ == "__main__":
    unittest.main()
