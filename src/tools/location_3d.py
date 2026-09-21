from src.tools.compat import Tool
import requests
import base64
from PIL import Image
import io
from src.utils.shared_memory import client_send_image
import numpy as np
import cv2
import torch
import os
from omegaconf import OmegaConf
from src.utils.coordinates import depth_boxes_to_habitat_world, detany_camera_to_habitat_world

class ObjectLocation3D(Tool):
    name = "ObjectLocation3D"
    description = (
        "Localize 3D objects and return exactly two values: box centers in Habitat world "
        "coordinates and their sizes in meters. If no matching object is detected, both values "
        "are empty lists. Use the exact image path returned by GoNextPointTool."
    )
    inputs = {
        "object": {"description": "the object that need to be localized", "type": "string"},
        "image_path": {
            "description": "List of the pathes to the images on which to localize 3D objects.",
            "type": "string",
        },
    }
    output_type = "any"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_id = kwargs.get("gpu_id", 0)
        self.debug = kwargs.get("debug", False)
        self.args = kwargs.get("args", None)
        self.navigation_tool = None
        self.last_coordinate_frame = None
        self.last_geometry_source = None
        self.last_resolved_image_paths = []
        if self.debug:
            return
        
        self.cfg = OmegaConf.load(self.args.cfg)
        OmegaConf.resolve(self.cfg)

        # Initialize any necessary components for 3D object localization here
        # For example, you might load a pre-trained model or set up a 3D environment
        self.endpoint = "location_3d"

    def bind_navigation_tool(self, navigation_tool):
        """Bind the observation source used to recover each image's camera pose."""
        self.navigation_tool = navigation_tool

    def _resolve_image_path(self, image_path: str) -> str:
        if self.navigation_tool is not None:
            return self.navigation_tool.resolve_image_path(image_path, fallback_to_current=True)
        candidates = [image_path]
        if not os.path.isabs(image_path):
            candidates.extend([os.path.join(self.cfg.output_dir, image_path), os.path.join(os.sep, image_path)])
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.realpath(os.path.abspath(candidate))
        raise FileNotFoundError(f"No usable image is available for {image_path!r}")

    def forward(self, object: str, image_path: str) -> list:
        if self.debug:
            return [[0, 0, 0]], [[1, 1, 1]]

        self.last_coordinate_frame = None
        self.last_geometry_source = None
        image_path = self._resolve_image_path(image_path)
        image = np.array(Image.open(image_path).convert("RGB"))
        image_path = os.path.realpath(os.path.abspath(image_path))
        self.last_resolved_image_paths = [image_path]

        data = {
            'endpoint': self.endpoint,
            'image': image,
            'text': object
        }
        res = client_send_image(data, self.gpu_id)

        if "error" in res.keys():
            error = str(res["error"])
            if "no object" in error.lower() or "not found" in error.lower():
                self.last_coordinate_frame = "habitat_world"
                self.last_geometry_source = "empty_detection"
                return [], []
            raise RuntimeError(f"DetAny3D failed for {object!r}: {error}")

        bboxes_3d = res["bboxes_3d"]
        if hasattr(bboxes_3d, "detach"):
            bboxes_3d = bboxes_3d.detach()
        if hasattr(bboxes_3d, "cpu"):
            bboxes_3d = bboxes_3d.cpu()
        if hasattr(bboxes_3d, "tolist"):
            bboxes_3d = bboxes_3d.tolist()
        if not bboxes_3d:
            self.last_coordinate_frame = "habitat_world"
            self.last_geometry_source = "empty_detection"
            return [], []
        camera_centers = [bbox[:3] for bbox in bboxes_3d]
        camera_pose = None
        if self.navigation_tool is not None:
            camera_pose = self.navigation_tool.camera_pose_for_image(image_path)
        if camera_pose is None:
            self.last_coordinate_frame = None
            raise RuntimeError(
                "No Habitat camera pose is registered for this image; refusing to expose "
                "DetAny3D camera-frame coordinates as world coordinates."
            )
        depth_image = None
        if self.navigation_tool is not None and hasattr(self.navigation_tool, "depth_for_image"):
            depth_image = self.navigation_tool.depth_for_image(image_path)

        depth_sizes = None
        if depth_image is not None:
            box_res = client_send_image(
                {
                    "endpoint": "location_2d",
                    "image": image,
                    "text": object,
                },
                self.gpu_id,
            )
            if "error" not in box_res:
                boxes_2d = box_res.get("bboxes_2d") or []
                if boxes_2d:
                    world_centers, depth_sizes = depth_boxes_to_habitat_world(
                        boxes_2d,
                        depth_image,
                        camera_pose,
                        float(self.cfg.hfov),
                    )
                    self.last_geometry_source = "habitat_depth+detany3d"
                else:
                    world_centers = detany_camera_to_habitat_world(camera_centers, camera_pose)
            else:
                error = str(box_res["error"])
                if "no object" in error.lower() or "not found" in error.lower():
                    world_centers = detany_camera_to_habitat_world(camera_centers, camera_pose)
                else:
                    raise RuntimeError(f"2D geometry anchoring failed for {object!r}: {error}")
        else:
            world_centers = detany_camera_to_habitat_world(camera_centers, camera_pose)

        if self.last_geometry_source is None:
            self.last_geometry_source = "detany3d_monocular"
        center = [[round(float(value), 2) for value in row] for row in world_centers]

        # DetAny3D decodes dimensions as [width, height, length].  The public
        # ToolEQA contract is [length, width, height].
        detany_sizes = [[bbox[5], bbox[3], bbox[4]] for bbox in bboxes_3d]
        if depth_sizes is not None and len(detany_sizes) != len(center):
            chosen_sizes = depth_sizes.tolist()
        else:
            chosen_sizes = detany_sizes[: len(center)]
        size = [[round(float(value), 2) for value in row] for row in chosen_sizes]
        # yaw = [bbox[6] for bbox in res["bboxes_3d"]]
        self.last_coordinate_frame = "habitat_world"

        return center, size

if __name__=="__main__":
    tool = ObjectLocation3D()
    import time
    t = time.time()
    result = tool.forward("light", "/mynvme1/EQA-Traj-0720/Z_HvYhe6T6e3msJ7HRsPAg/1-4.png")
    print("Time taken:", time.time() - t)
    print(np.array(result['rot_mat'][0]).shape)
