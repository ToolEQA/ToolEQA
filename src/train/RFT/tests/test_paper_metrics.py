from __future__ import annotations

import math
import unittest

from src.train.RFT.paper_metrics import compute_paper_metrics, weighted_recall


class PaperMetricsTest(unittest.TestCase):
    def test_weighted_recall_uses_paper_distance_and_fov(self) -> None:
        poses = [((0.0, 0.0, 0.0), 0.0)]
        # yaw=0 faces -z. The first target is 2 m ahead; the second is 90
        # degrees to the side and therefore outside a 120-degree FOV.
        targets = [(0.0, 0.0, -2.0), (2.0, 0.0, 0.0)]
        self.assertAlmostEqual(weighted_recall(poses, targets, 5.0), 0.3)

    def test_metrics_match_existing_table_normalization_and_raw_equation(self) -> None:
        sample = {
            "traj_length": 2.0,
            "evidence_targets": [{"name": "chair", "pos": [0.0, 0.0, -2.0]}],
        }
        trace = [
            {
                "action_type": "Navigate",
                "ok": True,
                "camera_state_after": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
                "path_length_after": 1.0,
            },
            {
                "action_type": "Navigate",
                "ok": True,
                "camera_state_after": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
                "path_length_after": 4.0,
            },
        ]
        metrics = compute_paper_metrics(sample, trace, correct=True)
        self.assertEqual(metrics["trajectory_steps"], 2.0)
        self.assertEqual(metrics["trajectory_length"], 4.0)
        self.assertAlmostEqual(metrics["raw_recall_at_5"], 0.6)
        self.assertAlmostEqual(metrics["recall_at_5"], 0.6 / math.sqrt(2.0))
        self.assertAlmostEqual(
            metrics["epath_at_5"],
            0.6 / math.sqrt(2.0) * math.exp(0.5),
        )

    def test_wrong_answer_zeroes_epath_but_not_recall(self) -> None:
        sample = {
            "traj_length": 1.0,
            "related_objects": [{"name": "chair", "pos": [0.0, 0.0, -1.0]}],
        }
        trace = [
            {
                "action_type": "Navigate",
                "ok": True,
                "camera_state_after": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
                "path_length_after": 1.0,
            }
        ]
        metrics = compute_paper_metrics(sample, trace, correct=False)
        self.assertGreater(metrics["recall_at_5"], 0.0)
        self.assertEqual(metrics["epath_at_5"], 0.0)


if __name__ == "__main__":
    unittest.main()
