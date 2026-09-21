from src.tools.compat import Tool
from src.runs.eqa_modeling import EQA_Modeling
from src.runs.go2_driver import Go2Driver
from omegaconf import OmegaConf
import cv2
import os
import numpy as np

class GoNextPointTool(Tool):
    name = "GoNextPointTool"
    description = "Continue toward the next exploration waypoint and obtain a new RGB observation."
    inputs = {
        "direction": {
            "description": "Next exploration direction. ONLY [`move_forward`, `turn_left`, `turn_right`, `turn_around`] are supported. The underlying Habitat navigation primitives match data generation: 0.25 meters forward and 30 degree left/right turns. One tool call may advance to a planner-selected waypoint using multiple primitives. `turn_around` reverses direction by 180 degrees.",
            "type": "string",
        },
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_id = kwargs.get("gpu_id", 0)
        self.debug = kwargs.get("debug", False)
        self.args = kwargs.get("args", None)
        self.real_robot = kwargs.get("real_robot", False)
        self.cfg = None
        self.eqa_modeling = None
        self._camera_poses = {}
        self._depth_images = {}

        if self.debug:
            return

        self.step_idx = -1

        self.cur_rgb_path = None

    def _ensure_backend(self):
        if self.debug or self.eqa_modeling is not None:
            return
        self.cfg = OmegaConf.load(self.args.cfg)
        OmegaConf.resolve(self.cfg)
        self.eqa_modeling = Go2Driver(self.cfg) if self.cfg.real_robot else EQA_Modeling(self.cfg, self.gpu_id)

    def initialize(self, data):
        self._ensure_backend()
        self.cur_rgb_path = self.eqa_modeling.initialize(data)
        self.sample_id = data['sample_id']
        self.save_dir = os.path.join(self.cfg.output_dir, self.sample_id)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.step_idx = 0
        self.cur_rgb_path = os.path.abspath(self.cur_rgb_path)
        self._camera_poses = {}
        self._depth_images = {}
        self._remember_current_camera_pose()

    @staticmethod
    def _canonical_path(image_path):
        return os.path.realpath(os.path.abspath(os.fspath(image_path)))

    def _remember_current_camera_pose(self):
        pose = getattr(self.eqa_modeling, "cam_pose_habitat", None)
        if self.cur_rgb_path and pose is not None:
            canonical = self._canonical_path(self.cur_rgb_path)
            self._camera_poses[canonical] = np.asarray(pose, dtype=float).copy()
            depth = getattr(self.eqa_modeling, "cur_depth", None)
            if depth is not None:
                self._depth_images[canonical] = np.asarray(depth, dtype=float).copy()

    def camera_pose_for_image(self, image_path):
        """Return the Habitat sensor-to-world pose used to capture an image."""
        if not image_path:
            return None
        candidates = [image_path]
        if self.cfg is not None and not os.path.isabs(os.fspath(image_path)):
            candidates.append(os.path.join(self.cfg.output_dir, os.fspath(image_path)))
        for candidate in candidates:
            pose = self._camera_poses.get(self._canonical_path(candidate))
            if pose is not None:
                return pose.copy()
        return None

    def depth_for_image(self, image_path):
        """Return the Habitat depth map captured with an episode RGB frame."""
        if not image_path:
            return None
        candidates = [image_path]
        if self.cfg is not None and not os.path.isabs(os.fspath(image_path)):
            candidates.append(os.path.join(self.cfg.output_dir, os.fspath(image_path)))
        for candidate in candidates:
            depth = self._depth_images.get(self._canonical_path(candidate))
            if depth is not None:
                return depth.copy()
        return None

    def resolve_image_path(self, image_path, *, fallback_to_current=True):
        """Resolve an image produced by the current episode.

        Rollouts for the same sample reuse one cache directory, so files left by
        an earlier rollout must not be treated as observations from the current
        simulator episode.  Prefer paths registered together with a camera pose
        and fall back to the current view when the controller names a stale or
        not-yet-created ``next_point`` image.
        """
        candidates = []
        if image_path:
            raw_path = os.fspath(image_path)
            candidates.append(raw_path)
            if not os.path.isabs(raw_path):
                if self.cfg is not None:
                    candidates.append(os.path.join(self.cfg.output_dir, raw_path))
                candidates.append(os.path.join(os.sep, raw_path))

        for candidate in candidates:
            canonical = self._canonical_path(candidate)
            if canonical in self._camera_poses and os.path.isfile(canonical):
                return canonical

        if fallback_to_current and self.cur_rgb_path:
            current = self._canonical_path(self.cur_rgb_path)
            if os.path.isfile(current):
                return current

        # Non-navigation derivatives such as crops have no camera pose.  They
        # remain valid image inputs, but only after current-episode candidates
        # and the safe current-view fallback have been considered.
        for candidate in candidates:
            canonical = self._canonical_path(candidate)
            if os.path.isfile(canonical):
                return canonical
        raise FileNotFoundError(f"No usable image is available for {image_path!r}")

    def register_derived_image(self, image_path, source_image_path):
        """Associate a crop/derived image with its source camera pose."""
        pose = self.camera_pose_for_image(source_image_path)
        if pose is not None:
            self._camera_poses[self._canonical_path(image_path)] = pose

    def forward(self, command=None, *, direction=None):
        """Execute one navigation request.

        ``direction`` is the public schema name exposed to the controller.
        Keep ``command`` and positional calls for compatibility with existing
        data-generation and inference code.
        """
        if direction is not None:
            if command is not None and command != direction:
                raise ValueError("Provide either command or direction, not conflicting values")
            command = direction
        if isinstance(command, dict):
            if 'direction' in command:
                command = command['direction']

        if command is None:
            raise ValueError("A navigation direction is required")

        if self.debug:
            return "./cache/init_rgb.png"
        self._ensure_backend()
        
        self.step_idx += 1
        save_path = os.path.join(self.save_dir, f"next_point_{self.step_idx}.jpg")
        self.eqa_modeling.go_next_point(command)
        if self.cfg.real_robot:
            cv2.imwrite(save_path, self.eqa_modeling.cur_rgb)
        else:
            cv2.imwrite(save_path, cv2.cvtColor(self.eqa_modeling.cur_rgb, cv2.COLOR_RGB2BGR))
        self.cur_rgb_path = os.path.abspath(save_path)
        self._remember_current_camera_pose()
        return self.cur_rgb_path
