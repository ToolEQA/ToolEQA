from __future__ import annotations

import unittest

from src.train.RFT.evidence import (
    EvidenceTracker,
    build_evidence_target_audit,
    evidence_target_issues,
    object_name_is_mentioned,
    object_names_match,
    unmentioned_related_objects,
)


class EvidenceTrackerTest(unittest.TestCase):
    def test_dataset_object_aliases_and_plurals_match_policy_queries(self) -> None:
        self.assertTrue(object_names_match("appliance", "refrigerator"))
        self.assertTrue(object_names_match("table", "kitchen island"))
        self.assertTrue(object_names_match("picture", "frames"))
        self.assertTrue(object_names_match("sofa", "couch"))
        self.assertTrue(object_names_match("tap", "faucet"))
        self.assertTrue(object_names_match("chair", "dining chairs"))
        self.assertTrue(object_names_match("chair", "leather recliner"))
        self.assertFalse(object_names_match("chair", "refrigerator"))
        self.assertFalse(object_names_match("refrigerator", "dishwasher"))

    def test_question_mentions_accept_dataset_aliases_but_reject_wrong_targets(self) -> None:
        self.assertTrue(object_name_is_mentioned("chair", "In which area is the black couch?"))
        self.assertTrue(object_name_is_mentioned("table", "Find the kitchen island in the room."))
        self.assertTrue(object_name_is_mentioned("rug", "Is the doormat lighter than the lamp?"))
        self.assertTrue(object_name_is_mentioned("clothes", "Are shoes next to the chair?"))
        self.assertFalse(object_name_is_mentioned("table", "Find the lamp next to a shelf."))

    def test_unmentioned_related_objects_reads_verl_extra_info(self) -> None:
        record = {
            "extra_info": {
                "question": "What is the distance between the chair and the lamp?",
                "related_objects": [{"name": "chair"}, {"name": "table"}],
            }
        }
        self.assertEqual(unmentioned_related_objects(record), ["table"])

    def test_pairwise_task_rejects_coincident_object_annotations(self) -> None:
        record = {
            "extra_info": {
                "question": "Which is larger, the sofa or the chair?",
                "question_type": "attribute-size",
                "related_objects": [
                    {"name": "sofa", "id": 1, "pos": [0.0, 0.0, 0.0]},
                    {"name": "chair", "id": 2, "pos": [0.005, 0.0, 0.0]},
                ],
            }
        }
        self.assertEqual(len(evidence_target_issues(record)), 1)

    def test_size_audit_rejects_context_object_left_after_question_rewrite(self) -> None:
        sample = {
            "question": "Which is larger, the chair or the lamp on the table?",
            "question_type": "attribute-size",
            "proposals": ["chair", "table", "same size", "unknown"],
            "answer": "B",
            "related_objects": [
                {"name": "chair", "id": 508, "pos": [0, 0, 0]},
                {"name": "table", "id": 506, "pos": [2, 0, 0]},
            ],
        }
        audit = build_evidence_target_audit(sample)
        self.assertFalse(audit["reward_eligible"])
        self.assertIn("nonprimary-target:1:table", audit["reward_audit"])

    def test_size_audit_accepts_ordered_comparison_operands(self) -> None:
        sample = {
            "question": "Which is larger, the chair or the table?",
            "question_type": "attribute-size",
            "proposals": ["chair", "table", "same size", "unknown"],
            "answer": "A",
            "related_objects": [
                {"name": "chair", "id": 1, "pos": [0, 0, 0]},
                {"name": "table", "id": 2, "pos": [2, 0, 0]},
            ],
        }
        audit = build_evidence_target_audit(sample)
        self.assertTrue(audit["reward_eligible"])
        self.assertEqual(audit["reward_audit"], [])
        self.assertEqual(audit["evidence_targets"], sample["related_objects"])

    def test_color_audit_rejects_context_object_as_operand(self) -> None:
        sample = {
            "question": "Which is more vibrant: the plant or the curtain beside the fridge?",
            "question_type": "attribute-color",
            "proposals": ["plant", "curtain", "same", "unknown"],
            "answer": "A",
            "related_objects": [
                {"name": "plant", "id": 1, "pos": [0, 0, 0]},
                {"name": "fridge", "id": 2, "pos": [2, 0, 0]},
            ],
        }
        audit = build_evidence_target_audit(sample)
        self.assertFalse(audit["reward_eligible"])
        self.assertIn("nonprimary-target:1:fridge", audit["reward_audit"])

    def test_explicit_ineligible_manifest_is_rejected_by_tracker(self) -> None:
        sample = {
            "question_type": "attribute-size",
            "reward_eligible": False,
            "reward_audit": ["nonprimary-target:1:table"],
            "evidence_targets": [{"name": "chair", "id": 1, "pos": [0, 0, 0]}],
        }
        with self.assertRaisesRegex(ValueError, "nonprimary-target"):
            EvidenceTracker(sample)

    def test_real_distance_type_requires_geometry(self) -> None:
        sample = {
            "question_type": "distance-distance",
            "related_objects": [{"name": "chair", "id": 1, "pos": [0, 0, 0]}],
        }
        tracker = EvidenceTracker(sample)
        self.assertEqual(tracker.spec.evidence_kind, "distance_relation")
        self.assertIn("chair#1:position", tracker.spec.required_facts)
        self.assertIn("task:distance_relation", tracker.spec.required_facts)

    def test_visual_evidence_requires_grounding_then_vqa(self) -> None:
        sample = {
            "question": "What color is the chair?",
            "question_type": "attribute-color",
            "related_objects": [{"name": "chair", "id": 1, "pos": [0, 0, 0]}],
        }
        trace = [
            {
                "action_type": "ObjectLocation2D",
                "args": {"object": "chair"},
                "result": {"bboxes_2d": [[1, 2, 20, 30]], "labels": ["chair"]},
                "ok": True,
                "image_path_before": "a.jpg",
            },
            {
                "action_type": "VisualQATool",
                "args": {"question": "What color is the chair?"},
                "result": "The chair is blue.",
                "ok": True,
                "image_path_before": "a.jpg",
            },
        ]
        report = EvidenceTracker(sample).replay(trace)
        self.assertEqual(report.coverage, 1.0)
        self.assertAlmostEqual(report.steps[0].delta, 1 / 3)
        self.assertAlmostEqual(report.steps[1].delta, 2 / 3)
        self.assertIn("task:visual_comparison", report.satisfied_facts)

    def test_vqa_without_grounding_is_not_evidence(self) -> None:
        sample = {
            "question": "What color is the chair?",
            "question_type": "attribute-color",
            "related_objects": [{"name": "chair", "id": 1, "pos": [0, 0, 0]}],
        }
        trace = [
            {
                "action_type": "VisualQA",
                "args": {"question": "What color is the chair?"},
                "result": "blue",
                "ok": True,
            }
        ]
        report = EvidenceTracker(sample).replay(trace)
        self.assertEqual(report.coverage, 0.0)
        self.assertEqual(report.no_progress_count, 1)

    def test_vqa_on_different_image_is_not_evidence(self) -> None:
        sample = {
            "question": "What color is the chair?",
            "question_type": "attribute-color",
            "related_objects": [{"name": "chair", "id": 1, "pos": [0, 0, 0]}],
        }
        trace = [
            {
                "action_type": "ObjectLocation2D",
                "args": {"object": "chair", "image_path": "a.jpg"},
                "result": {"bboxes_2d": [[1, 2, 20, 30]], "labels": ["chair"], "scores": [0.8]},
                "ok": True,
            },
            {
                "action_type": "VisualQATool",
                "args": {"question": "What color is the chair?", "image_path": "b.jpg"},
                "result": "The chair is blue.",
                "ok": True,
            },
        ]
        report = EvidenceTracker(sample).replay(trace)
        self.assertEqual(report.coverage, 1 / 3)
        self.assertTrue(report.steps[1].no_progress)

    def test_vqa_on_crop_inherits_grounding_from_source_view(self) -> None:
        sample = {
            "question": "What color is the chair?",
            "question_type": "attribute-special",
            "related_objects": [{"name": "chair", "id": 1, "pos": [0, 0, 0]}],
        }
        trace = [
            {
                "action_type": "ObjectLocation2D",
                "args": {"object": "chair", "image_path": "room.jpg"},
                "result": {"bboxes_2d": [[1, 2, 20, 30]], "labels": ["chair"], "scores": [0.9]},
                "ok": True,
            },
            {
                "action_type": "ObjectCrop",
                "args": {"bounding_box": [1, 2, 20, 30], "image_path": "room.jpg"},
                "result": ["chair_crop.jpg"],
                "ok": True,
            },
            {
                "action_type": "VisualQA",
                "args": {"question": "What color is the chair?", "image_paths": ["chair_crop.jpg"]},
                "result": "blue",
                "ok": True,
            },
        ]
        report = EvidenceTracker(sample).replay(trace)
        self.assertEqual(report.coverage, 1.0)
        self.assertFalse(report.steps[1].invalid)
        self.assertIn("task:attribute_resolution", report.satisfied_facts)

    def test_low_confidence_grounding_is_rejected(self) -> None:
        sample = {
            "question_type": "attribute-color",
            "related_objects": [{"name": "chair", "id": 1, "pos": [0, 0, 0]}],
        }
        trace = [
            {
                "action_type": "ObjectLocation2D",
                "args": {"object": "chair", "image_path": "a.jpg"},
                "result": {"bboxes_2d": [[1, 2, 20, 30]], "labels": ["chair"], "scores": [0.2]},
                "ok": True,
            }
        ]
        report = EvidenceTracker(sample).replay(trace)
        self.assertEqual(report.coverage, 0.0)
        self.assertEqual(report.invalid_count, 1)

    def test_3d_result_must_match_ground_truth_position_for_geometry(self) -> None:
        sample = {
            "question_type": "distance",
            "related_objects": [{"name": "chair", "id": 1, "pos": [0, 0, 0]}],
        }
        trace = [
            {
                "action_type": "Location3D",
                "args": {"object": "chair"},
                "result": [[10, 10, 10], [1, 1, 1]],
                "ok": True,
            }
        ]
        report = EvidenceTracker(sample, position_tolerance=1.0).replay(trace)
        self.assertEqual(report.coverage, 1 / 3)
        self.assertIn("chair#1:grounded", report.satisfied_facts)
        self.assertNotIn("chair#1:position", report.satisfied_facts)
        self.assertEqual(report.invalid_count, 0)

    def test_3d_result_within_tolerance_adds_geometry(self) -> None:
        sample = {
            "question_type": "distance",
            "related_objects": [{"name": "stair", "id": 219, "pos": [-8.155, 2.385, -5.038]}],
        }
        trace = [
            {
                "action_type": "Location3D",
                "args": {"object": "stair", "image_path": "view.jpg"},
                "result": [[[-7.43, 3.0, -6.42]], [[0.42, 0.29, 0.33]]],
                "coordinate_frame": "habitat_world",
                "ok": True,
            }
        ]
        report = EvidenceTracker(sample, position_tolerance=2.0).replay(trace)
        self.assertEqual(report.coverage, 1.0)
        self.assertIn("stair#219:position", report.satisfied_facts)
        self.assertIn("task:distance_relation", report.satisfied_facts)
        self.assertEqual(report.measurements["stair#219"]["size"], [0.42, 0.29, 0.33])

    def test_size_comparison_requires_all_3d_operands(self) -> None:
        sample = {
            "question_type": "attribute-size",
            "related_objects": [
                {"name": "chair", "id": 1, "pos": [0, 0, 0]},
                {"name": "table", "id": 2, "pos": [2, 0, 0]},
            ],
        }
        trace = [
            {
                "action_type": "Location3D",
                "args": {"object": "chair", "image_path": "chair.jpg"},
                "result": [[[0, 0, 0]], [[1, 2, 3]]],
                "coordinate_frame": "habitat_world",
                "ok": True,
            },
            {
                "action_type": "Location3D",
                "args": {"object": "table", "image_path": "table.jpg"},
                "result": [[[2, 0, 0]], [[2, 2, 2]]],
                "coordinate_frame": "habitat_world",
                "ok": True,
            },
        ]
        report = EvidenceTracker(sample).replay(trace)
        self.assertEqual(report.coverage, 1.0)
        self.assertIn("task:geometry_comparison", report.satisfied_facts)
        objects = report.derived_evidence["objects"]
        self.assertEqual([item["volume"] for item in objects], [6.0, 8.0])

    def test_distance_relation_exposes_verified_pairwise_distance(self) -> None:
        sample = {
            "question_type": "distance-distance",
            "related_objects": [
                {"name": "chair", "id": 1, "pos": [0, 0, 0]},
                {"name": "table", "id": 2, "pos": [3, 4, 0]},
            ],
        }
        trace = [
            {
                "action_type": "Location3D",
                "args": {"object": "chair", "image_path": "chair.jpg"},
                "result": [[[0, 0, 0]], [[1, 1, 1]]],
                "coordinate_frame": "habitat_world",
                "ok": True,
            },
            {
                "action_type": "Location3D",
                "args": {"object": "table", "image_path": "table.jpg"},
                "result": [[[3, 4, 0]], [[1, 1, 1]]],
                "coordinate_frame": "habitat_world",
                "ok": True,
            },
        ]
        report = EvidenceTracker(sample).replay(trace)
        self.assertEqual(report.coverage, 1.0)
        self.assertEqual(report.derived_evidence["pairwise_distances"][0]["distance"], 5.0)

    def test_counting_requires_unique_3d_instances(self) -> None:
        sample = {
            "question_type": "counting-counting",
            "related_objects": [
                {"name": "bed", "id": 1, "pos": [0, 0, 0]},
                {"name": "bed", "id": 2, "pos": [3, 0, 0]},
            ],
        }
        trace = [
            {
                "action_type": "Location3D",
                "args": {"object": "bed", "image_path": "beds.jpg"},
                "result": [[[0, 0, 0], [3, 0, 0]], [[1, 1, 1], [1, 1, 1]]],
                "coordinate_frame": "habitat_world",
                "ok": True,
            }
        ]
        report = EvidenceTracker(sample).replay(trace)
        self.assertEqual(report.coverage, 1.0)
        self.assertEqual(report.derived_evidence["verified_count"], 2)
        self.assertIn("task:instance_count", report.satisfied_facts)

    def test_semantic_location_requires_position_and_visual_observation(self) -> None:
        sample = {
            "question": "In which room is the oven located?",
            "question_type": "location-location",
            "related_objects": [{"name": "oven", "id": 1, "pos": [0, 0, 0], "region_id": "kitchen"}],
        }
        locate = {
            "action_type": "Location3D",
            "args": {"object": "oven", "image_path": "oven.jpg"},
            "result": [[[0, 0, 0]], [[1, 1, 1]]],
            "coordinate_frame": "habitat_world",
            "ok": True,
        }
        partial = EvidenceTracker(sample).replay([locate])
        self.assertLess(partial.coverage, 1.0)
        vqa = {
            "action_type": "VisualQA",
            "args": {"question": "In which room is the oven located?", "image_path": "oven.jpg"},
            "result": "The oven is in the kitchen.",
            "ok": True,
        }
        complete = EvidenceTracker(sample).replay([locate, vqa])
        self.assertEqual(complete.coverage, 1.0)
        self.assertIn("task:room_localization", complete.satisfied_facts)

    def test_3d_result_in_camera_frame_is_rejected(self) -> None:
        sample = {
            "question_type": "distance",
            "related_objects": [{"name": "chair", "id": 1, "pos": [0, 0, 0]}],
        }
        trace = [
            {
                "action_type": "Location3D",
                "args": {"object": "chair"},
                "result": [[[0, 0, 0]], [[1, 1, 1]]],
                "coordinate_frame": "detany_camera",
                "ok": True,
            }
        ]
        report = EvidenceTracker(sample).replay(trace)
        self.assertEqual(report.coverage, 0.0)
        self.assertEqual(report.invalid_count, 1)

    def test_empty_3d_detection_is_no_progress_not_invalid(self) -> None:
        sample = {
            "question_type": "distance",
            "related_objects": [{"name": "chair", "id": 1, "pos": [0, 0, 0]}],
        }
        trace = [
            {
                "action_type": "Location3D",
                "args": {"object": "chair", "image_path": "view.jpg"},
                "result": [[], []],
                "coordinate_frame": "habitat_world",
                "ok": True,
            }
        ]
        report = EvidenceTracker(sample).replay(trace)
        self.assertEqual(report.coverage, 0.0)
        self.assertEqual(report.invalid_count, 0)
        self.assertEqual(report.no_progress_count, 1)

    def test_navigation_same_command_new_view_is_not_duplicate(self) -> None:
        sample = {"question_type": "unknown", "related_objects": []}
        trace = [
            {"action_type": "Navigate", "args": {"direction": "move_forward"}, "image_path_before": "a"},
            {"action_type": "Navigate", "args": {"direction": "move_forward"}, "image_path_before": "b"},
        ]
        self.assertEqual(EvidenceTracker(sample).replay(trace).duplicate_count, 0)


if __name__ == "__main__":
    unittest.main()
