from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from src.tools.crop import ObjectCrop
from src.tools.go_next_point import GoNextPointTool
from src.tools.location_2d import ObjectLocation2D
from src.tools.location_3d import ObjectLocation3D
from src.train.RFT.verl_adapter.agent_loop import (
    DuplicateComputeActionError,
    DuplicateToolCallError,
    ToolEQAEvidenceAgentLoop,
    _RecordedTool,
    _ToolExecutionRecorder,
)


class RolloutRobustnessTest(unittest.TestCase):
    def test_recorder_captures_navigation_camera_pose(self) -> None:
        backend = SimpleNamespace(
            agent_state=SimpleNamespace(position=np.array([1.0, 2.0, 3.0])),
            angle=0.75,
            path_length=4.0,
        )
        navigation = SimpleNamespace(
            name="GoNextPointTool",
            eqa_modeling=backend,
            cur_rgb_path="view.jpg",
        )
        recorder = _ToolExecutionRecorder([navigation])
        self.assertEqual(
            recorder.current_camera_state(),
            {"position": [1.0, 2.0, 3.0], "yaw": 0.75},
        )

    def test_identical_perception_call_is_rejected_before_reexecution(self) -> None:
        class FakeTool:
            name = "ObjectLocation2D"

            def __init__(self):
                self.calls = 0

            def __call__(self, *args, **kwargs):
                self.calls += 1
                return {"bboxes_2d": [[1, 2, 3, 4]]}

        tool = FakeTool()
        recorder = _ToolExecutionRecorder([tool])
        wrapped = _RecordedTool(tool.name, tool, recorder)
        first = wrapped(object="chair", image_path="view.jpg")
        self.assertEqual(first["bboxes_2d"], [[1, 2, 3, 4]])
        with self.assertRaises(DuplicateToolCallError):
            wrapped(object="chair", image_path="view.jpg")
        self.assertEqual(tool.calls, 1)
        self.assertTrue(recorder.trace[-1]["duplicate_rejected"])
        self.assertTrue(recorder.memory.action_history[-1]["duplicate_rejected"])

    def test_navigation_accepts_public_direction_keyword(self) -> None:
        navigation = GoNextPointTool(debug=True)
        self.assertEqual(navigation(direction="turn_left"), "./cache/init_rgb.png")
        self.assertEqual(navigation("move_forward"), "./cache/init_rgb.png")

    def test_stale_rollout_frame_falls_back_to_current_episode_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            current = os.path.join(directory, "next_point_1.jpg")
            stale = os.path.join(directory, "next_point_9.jpg")
            Image.new("RGB", (4, 4)).save(current)
            Image.new("RGB", (4, 4)).save(stale)

            navigation = GoNextPointTool.__new__(GoNextPointTool)
            navigation.cfg = SimpleNamespace(output_dir=directory)
            navigation.cur_rgb_path = current
            navigation._camera_poses = {
                navigation._canonical_path(current): np.eye(4),
            }

            self.assertEqual(navigation.resolve_image_path(stale), navigation._canonical_path(current))

    def test_registered_prior_frame_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            current = os.path.join(directory, "next_point_2.jpg")
            prior = os.path.join(directory, "next_point_1.jpg")
            Image.new("RGB", (4, 4)).save(current)
            Image.new("RGB", (4, 4)).save(prior)

            navigation = GoNextPointTool.__new__(GoNextPointTool)
            navigation.cfg = SimpleNamespace(output_dir=directory)
            navigation.cur_rgb_path = current
            navigation._camera_poses = {
                navigation._canonical_path(current): np.eye(4),
                navigation._canonical_path(prior): np.eye(4),
            }

            self.assertEqual(navigation.resolve_image_path(prior), navigation._canonical_path(prior))

    def test_detany_no_detection_keeps_two_value_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "view.jpg")
            Image.new("RGB", (4, 4)).save(image_path)

            tool = ObjectLocation3D.__new__(ObjectLocation3D)
            tool.debug = False
            tool.cfg = SimpleNamespace(output_dir=directory)
            tool.navigation_tool = None
            tool.endpoint = "location_3d"
            tool.gpu_id = 0
            tool.last_coordinate_frame = None
            tool.last_resolved_image_paths = []

            with patch(
                "src.tools.location_3d.client_send_image",
                return_value={"error": "No objects found in the image."},
            ):
                centers, sizes = tool.forward("chair", image_path)

            self.assertEqual(centers, [])
            self.assertEqual(sizes, [])
            self.assertEqual(tool.last_coordinate_frame, "habitat_world")

    def test_detany_runtime_failure_raises_instead_of_changing_arity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "view.jpg")
            Image.new("RGB", (4, 4)).save(image_path)

            tool = ObjectLocation3D.__new__(ObjectLocation3D)
            tool.debug = False
            tool.cfg = SimpleNamespace(output_dir=directory)
            tool.navigation_tool = None
            tool.endpoint = "location_3d"
            tool.gpu_id = 0
            tool.last_coordinate_frame = None
            tool.last_resolved_image_paths = []

            with patch(
                "src.tools.location_3d.client_send_image",
                return_value={"error": "shared-memory server disconnected"},
            ):
                with self.assertRaisesRegex(RuntimeError, "DetAny3D failed"):
                    tool.forward("chair", image_path)

    def test_3d_localization_anchors_center_to_habitat_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "view.jpg")
            Image.new("RGB", (4, 4)).save(image_path)
            navigation = Mock()
            navigation.resolve_image_path.return_value = image_path
            navigation.camera_pose_for_image.return_value = np.eye(4)
            navigation.depth_for_image.return_value = np.full((4, 4), 2.0)

            tool = ObjectLocation3D.__new__(ObjectLocation3D)
            tool.debug = False
            tool.cfg = SimpleNamespace(output_dir=directory, hfov=90.0)
            tool.navigation_tool = navigation
            tool.endpoint = "location_3d"
            tool.gpu_id = 0
            tool.last_coordinate_frame = None
            tool.last_geometry_source = None
            tool.last_resolved_image_paths = []

            with patch(
                "src.tools.location_3d.client_send_image",
                side_effect=[
                    {"bboxes_3d": np.array([[0.0, 0.0, 9.0, 1.0, 2.0, 3.0, 0.0]])},
                    {"bboxes_2d": [[0, 0, 4, 4]]},
                ],
            ):
                centers, sizes = tool.forward("chair", image_path)

            self.assertEqual(centers, [[0.0, 0.0, -2.0]])
            self.assertEqual(sizes, [[3.0, 1.0, 2.0]])
            self.assertEqual(tool.last_coordinate_frame, "habitat_world")
            self.assertEqual(tool.last_geometry_source, "habitat_depth+detany3d")

    def test_2d_no_detection_is_a_valid_empty_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "view.jpg")
            Image.new("RGB", (8, 6)).save(image_path)

            tool = ObjectLocation2D.__new__(ObjectLocation2D)
            tool.debug = False
            tool.cfg = SimpleNamespace(output_dir=directory)
            tool.navigation_tool = None
            tool.endpoint = "location_2d"
            tool.gpu_id = 0
            tool.last_resolved_image_paths = []

            with patch(
                "src.tools.location_2d.client_send_image",
                return_value={"error": "No objects found in the image."},
            ):
                result = tool.forward("chair", image_path)

            self.assertEqual(result["bboxes_2d"], [])
            self.assertEqual(result["image_size"], [8, 6])

    def test_crop_exposes_canonical_argument_and_accepts_legacy_alias(self) -> None:
        self.assertIn("bounding_box", ObjectCrop.inputs)
        self.assertNotIn("bound_boxes", ObjectCrop.inputs)
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "view.jpg")
            Image.new("RGB", (8, 6)).save(image_path)
            tool = ObjectCrop.__new__(ObjectCrop)
            tool.debug = False
            tool.navigation_tool = None
            tool.last_resolved_image_paths = []
            output = tool.forward(bound_boxes=[0, 0, 4, 3], image_path=image_path)
            self.assertEqual(len(output), 1)
            self.assertTrue(os.path.isfile(output[0]))

    def test_forced_final_recovers_letter_or_proposal(self) -> None:
        sample = {"proposals": ["wood", "metal", "glass", "plastic"]}
        self.assertEqual(
            ToolEQAEvidenceAgentLoop._best_effort_final_answer("The best option is C.", sample),
            "C",
        )
        self.assertEqual(
            ToolEQAEvidenceAgentLoop._best_effort_final_answer("It appears to be metal.", sample),
            "B",
        )

    def test_rft_prompt_declares_single_pass_thought_code_and_choice_letter(self) -> None:
        loop = ToolEQAEvidenceAgentLoop.__new__(ToolEQAEvidenceAgentLoop)
        root = Path(__file__).resolve().parents[4]
        loop.react_system_prompt_template = (
            root / "data/ToolTrajectory/prompts/rft_thought_code_system_prompt.txt"
        ).read_text(encoding="utf-8")
        prompt = loop._build_system_prompt([])
        self.assertIn("exactly one concise `Thought:`", prompt)
        self.assertIn("exactly one executable `Code:` block", prompt)
        self.assertIn("will not be present on the next turn", prompt)
        self.assertIn("Only the resulting Observation and Spatial Memory persist", prompt)
        self.assertIn("one local Compute action per turn", prompt)
        self.assertIn("cannot create new perceptual evidence", prompt)
        self.assertIn('final_answer("B")', prompt)
        self.assertIn("Pass exactly one uppercase letter", prompt)
        self.assertNotIn("<think>", prompt)
        self.assertNotIn("2048-token limit", prompt)
        self.assertNotIn("<<", prompt)

    def test_agent_config_uses_one_bounded_thought_code_generation(self) -> None:
        root = Path(__file__).resolve().parents[4]
        agent_config = (
            root / "src/train/RFT/verl_adapter/configs/agent_loop.yaml"
        ).read_text(encoding="utf-8")
        rollout_config = (
            root / "src/train/RFT/verl_adapter/configs/evidence_grpo.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("normal_turn_max_tokens: 768", agent_config)
        self.assertNotIn("thought_max_tokens", agent_config)
        self.assertNotIn("code_max_tokens", agent_config)
        self.assertIn("Qwen3-VL-8B-Instruct", rollout_config)
        self.assertIn("max_model_len: 16384", rollout_config)

    def test_lazy_import_cannot_overwrite_parameterized_agent_registry(self) -> None:
        root = Path(__file__).resolve().parents[4]
        source = (
            root / "src/train/RFT/verl_adapter/agent_loop.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('@register("tooleqa_evidence_agent")', source)
        self.assertIn('kwargs.pop("normal_turn_max_tokens", 768)', source)
        self.assertNotIn('kwargs.pop("thought_max_tokens"', source)
        self.assertNotIn('kwargs.pop("code_max_tokens"', source)

    def test_missing_print_recovers_the_tool_result(self) -> None:
        recovered = ToolEQAEvidenceAgentLoop._recover_unprinted_tool_result(
            ToolEQAEvidenceAgentLoop._no_print_observation,
            [{"ok": True, "result": {"bboxes_2d": [[1, 2, 3, 4]]}}],
        )
        self.assertIn('"bboxes_2d": [[1, 2, 3, 4]]', recovered)

    def test_tool_free_python_is_recorded_as_compute(self) -> None:
        loop = ToolEQAEvidenceAgentLoop.__new__(ToolEQAEvidenceAgentLoop)
        recorder = _ToolExecutionRecorder([])
        state: dict = {"chair_position": [0, 0, 0], "table_position": [3, 4, 0]}

        observation = loop._execute_turn(
            "import math\ndistance = math.dist(chair_position, table_position)\nprint(distance)",
            {},
            state,
            recorder,
        )

        self.assertEqual(observation, "5.0")
        self.assertEqual(state["distance"], 5.0)
        self.assertEqual(recorder.trace[-1]["action_type"], "Compute")
        self.assertTrue(recorder.trace[-1]["ok"])
        self.assertIn("distance", recorder.trace[-1]["state_changes"])
        self.assertEqual(recorder.memory.action_history[-1]["tool"], "Compute")

    def test_identical_compute_with_identical_inputs_is_rejected(self) -> None:
        loop = ToolEQAEvidenceAgentLoop.__new__(ToolEQAEvidenceAgentLoop)
        recorder = _ToolExecutionRecorder([])
        state = {"values": [1, 2, 3]}
        code = "total = sum(values)\nprint(total)"

        self.assertEqual(loop._execute_turn(code, {}, state, recorder), "6")
        rejected = loop._execute_turn(code, {}, state, recorder)

        self.assertIn("Identical Compute action rejected", rejected)
        self.assertEqual(len(recorder.trace), 2)
        self.assertEqual(recorder.trace[-1]["action_type"], "Compute")
        self.assertFalse(recorder.trace[-1]["ok"])
        self.assertTrue(recorder.trace[-1]["duplicate_rejected"])
        self.assertIn("DuplicateComputeActionError", recorder.trace[-1]["error"])

    def test_compute_without_print_persists_a_synthesized_observation(self) -> None:
        loop = ToolEQAEvidenceAgentLoop.__new__(ToolEQAEvidenceAgentLoop)
        recorder = _ToolExecutionRecorder([])
        state: dict = {}

        observation = loop._execute_turn("answer = 6 * 7", {}, state, recorder)

        self.assertIn('"answer": 42', observation)
        self.assertEqual(recorder.trace[-1]["action_type"], "Compute")

    def test_true_python_noop_remains_invalid(self) -> None:
        loop = ToolEQAEvidenceAgentLoop.__new__(ToolEQAEvidenceAgentLoop)
        recorder = _ToolExecutionRecorder([])

        observation = loop._execute_turn("2 + 2", {}, {}, recorder)

        self.assertIn("produced no tool call, printed output, or state change", observation)
        self.assertEqual(recorder.trace[-1]["action_type"], "InvalidCode")
        self.assertEqual(recorder.memory.action_history[-1]["tool"], "InvalidCode")


if __name__ == "__main__":
    unittest.main()
