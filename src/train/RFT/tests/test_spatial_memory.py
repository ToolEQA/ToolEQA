from __future__ import annotations

import unittest

from src.memory.spatial_memory import SpatialMemory


class SpatialMemoryInstanceTest(unittest.TestCase):
    def test_reward_targets_are_not_exposed_to_policy_memory(self) -> None:
        memory = SpatialMemory()
        memory.configure_task(
            {
                "question_type": "attribute-color",
                "evidence_targets": [{"name": "privileged-chair"}],
                "related_objects": [{"name": "privileged-chair"}],
            }
        )
        self.assertEqual(memory.task_targets, [])
        self.assertNotIn("privileged-chair", memory.serialize())

    def test_task_checklist_and_action_history_are_serialized(self) -> None:
        memory = SpatialMemory()
        memory.configure_task(
            {
                "question_type": "attribute-color",
                "policy_targets": [{"name": "chair"}],
            }
        )
        memory.update(
            "ObjectLocation2D",
            {"object": "chair", "image_path": "view.jpg"},
            {"bboxes_2d": [[1, 2, 3, 4]]},
            0,
        )
        memory.record_action(
            "ObjectLocation2D",
            {"object": "chair", "image_path": "view.jpg"},
            {"bboxes_2d": [[1, 2, 3, 4]]},
            0,
            ok=True,
        )
        rendered = memory.serialize()
        self.assertIn("Evidence checklist", rendered)
        self.assertIn("grounded=done", rendered)
        self.assertIn("visual=missing", rendered)
        self.assertIn("Recent actions", rendered)
        self.assertIn("Recommended next evidence actions", rendered)

    def test_multiple_same_category_detections_are_kept_as_instances(self) -> None:
        memory = SpatialMemory()
        memory.update(
            "ObjectLocation3D",
            {"object": "bed", "image_path": "beds.jpg"},
            ([[0, 0, 0], [3, 0, 0]], [[1, 2, 1], [1, 2, 1]]),
            0,
        )
        self.assertEqual(set(memory.detected_objects), {"bed", "bed#2"})
        self.assertEqual(memory.detected_objects["bed#2"]["position"], [3.0, 0.0, 0.0])

    def test_repeat_world_detection_updates_instead_of_double_counting(self) -> None:
        memory = SpatialMemory()
        memory.update(
            "ObjectLocation3D",
            {"object": "bed", "image_path": "first.jpg"},
            ([[0, 0, 0]], [[1, 2, 1]]),
            0,
        )
        memory.update(
            "ObjectLocation3D",
            {"object": "bed", "image_path": "second.jpg"},
            ([[0.2, 0, 0.1]], [[1, 2, 1]]),
            1,
        )
        self.assertEqual(list(memory.detected_objects), ["bed"])
        self.assertEqual(memory.detected_objects["bed"]["step"], 1)

    def test_distinct_world_detection_creates_a_new_instance(self) -> None:
        memory = SpatialMemory()
        memory.update("ObjectLocation3D", {"object": "chair"}, ([[0, 0, 0]], [[1, 1, 1]]), 0)
        memory.update("ObjectLocation3D", {"object": "chair"}, ([[2, 0, 0]], [[1, 1, 1]]), 1)
        self.assertEqual(set(memory.detected_objects), {"chair", "chair#2"})


if __name__ == "__main__":
    unittest.main()
