import math
import unittest
from pathlib import Path
from unittest.mock import patch

from src.train.RFT.open_protocol import open_system_prompt, semantic_judgment
from src.train.RFT.prepare_open import convert
from src.train.RFT.reward import compute_reward
from src.train.RFT.paper_metrics import compute_paper_metrics
from src.train.RFT.tests.test_reward import SAMPLE, location, final


class OpenProtocolTest(unittest.TestCase):
    def test_no_choices_or_stored_plan_at_development(self):
        record = {"extra_info": {**SAMPLE, "sample_id": "test", "plan": "privileged plan"},
                  "prompt": [], "reward_model": {}}
        result = convert(record)
        self.assertNotIn("proposals", result["extra_info"])
        self.assertNotIn("plan", result["extra_info"])
        self.assertEqual(result["extra_info"]["answer"], "yes")
        self.assertEqual(result["prompt"][0]["content"], SAMPLE["question"])
        self.assertIn("proposals", record["extra_info"])

    def test_option_dependent_reference_rejected(self):
        record = {"extra_info": {**SAMPLE, "sample_id": "x", "proposals": ["All of the above"]}}
        with self.assertRaises(ValueError):
            convert(record)

    def test_prompt_has_natural_answer_not_option_instruction(self):
        prompt = open_system_prompt(Path("data/ToolTrajectory/prompts/rft_thought_code_system_prompt.txt").read_text())
        self.assertNotIn("choices A", prompt)
        self.assertNotIn('final_answer("B")', prompt)
        self.assertNotIn("uppercase letter", prompt)

    def test_graded_gated_reward_and_missing_penalty(self):
        sample = {**SAMPLE, "answer_setting": "open", "answer": "yes"}
        trace = [location("chair", [0, 0, 0], 0), location("table", [1, 0, 0], 1), final("yes")]
        for sigma in range(6):
            r = compute_reward(sample, trace, semantic_score=sigma)
            self.assertAlmostEqual(r["answer_objective"], 2*sigma/5-1)
            self.assertEqual(r["answer_quality"], sigma/5)
        self.assertEqual(compute_reward(sample, [], semantic_score=0)["answer_objective"], -2)
        self.assertEqual(compute_reward(sample, [final("yes")], semantic_score=5)["answer_objective"], 0)
        self.assertEqual(compute_reward(sample, trace, semantic_score=5,
                                       weights={"reward_phase": "evidence"})["answer_objective"], 0)
        with self.assertRaises(ValueError):
            compute_reward(sample, trace)

    def test_open_metrics_use_graded_quality_and_paper_recall(self):
        sample = {"related_objects": [{"pos": [0, 0, -1]}], "traj_length": 2}
        trace = [{"action_type": "Navigate", "camera_state_after": {"position": [0, 0, 0], "yaw": 0},
                  "path_length_after": 2}] * 4
        result = compute_paper_metrics(sample, trace, correct=False, answer_quality=.6)
        self.assertAlmostEqual(result["recall_at_5"], .8)
        self.assertAlmostEqual(result["epath_at_5"], .6*.8*math.e)
        self.assertAlmostEqual(result["legacy_recall_at_5"], .4)

    def test_blank_and_letter_score_zero_without_service(self):
        with patch("src.train.RFT.open_protocol.request", side_effect=AssertionError("unexpected call")):
            for value in ("", "A", "b."):
                self.assertEqual(semantic_judgment("q", "yes", value)["score"], 0)

    def test_bad_judge_output_fails_closed(self):
        with patch("src.train.RFT.open_protocol.request", return_value={"score": 6}):
            with self.assertRaises(ValueError):
                semantic_judgment("q", "yes", "no")

    def test_policy_builder_never_uses_open_options_or_test_reference_plan(self):
        from src.train.RFT.verl_adapter.agent_loop import ToolEQAEvidenceAgentLoop
        sample = {**SAMPLE, "answer_setting": "open", "plan": "SECRET",
                  "planner_source": "question-only-frozen-v1"}
        with patch("src.train.RFT.open_protocol.question_only_plan", return_value="PUBLIC PLAN") as planner:
            task = ToolEQAEvidenceAgentLoop._build_task([], sample)
        planner.assert_called_once_with(SAMPLE["question"])
        self.assertNotIn("SECRET", task)
        self.assertNotIn("Choices:", task)
        self.assertIn("PUBLIC PLAN", task)
        self.assertEqual(ToolEQAEvidenceAgentLoop._best_effort_final_answer("Thought: guess A", sample), "")


if __name__ == "__main__":
    unittest.main()
