import unittest
from unittest.mock import patch

from src.evaluation.official_eval import compute_score
from src.evaluation.summarize_rollouts import FIELDS


class FullMetricsTest(unittest.TestCase):
    def test_full_split_keeps_ineligible_sample_and_reports_evidence(self):
        sample = {"question": "What color is the chair?", "answer": "red",
                  "question_type": "attribute-special", "answer_setting": "open",
                  "related_objects": [{"name": "chair", "id": 1, "pos": [0, 0, 0]}],
                  "reward_eligible": False, "reward_audit": ["training-only filter"],
                  "evidence_targets": []}
        with patch("src.evaluation.open_protocol.semantic_judgment",
                   return_value={"score": 1, "answer_quality": 0.0}), \
                patch("src.evaluation.official_eval.attach_reward"):
            result = compute_score("", "", "red", {
                "sample_info": sample, "tooleqa_trace": [], "tooleqa_final_answer": "blue"})
        self.assertEqual(result["semantic_score"], 1)
        self.assertEqual(result["evidence_coverage"], 0.0)
        self.assertEqual(result["evidence_complete"], 0.0)
        self.assertFalse(sample["reward_eligible"])
        self.assertIn("evidence_complete", FIELDS)


if __name__ == "__main__":
    unittest.main()
