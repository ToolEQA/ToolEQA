from src.tools.compat import Tool
import os
import json
from PIL import Image

class ObjectCrop(Tool):
    name = "ObjectCrop"
    description = "Given the bounding boxes of objects, crop and save the relevant objects from the image."
    inputs = {
        "bounding_box": {
            "description": "One [x1, y1, x2, y2] box or a list of such boxes.",
            "type": "any",
        },
        "image_path": {
            "description": "The path to the image on which to crop objects.",
            "type": "string",
        },
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_id = kwargs.get("gpu_id", 0)
        self.debug = kwargs.get("debug", False)
        self.image_root = "data/EQA-Traj-0720"
        self.navigation_tool = None
        self.last_resolved_image_paths = []
        if self.debug:
            return

    def bind_navigation_tool(self, navigation_tool):
        self.navigation_tool = navigation_tool

    def _resolve_image_path(self, image_path: str) -> str:
        if self.navigation_tool is not None:
            return self.navigation_tool.resolve_image_path(image_path, fallback_to_current=True)
        if os.path.isfile(image_path):
            return os.path.realpath(os.path.abspath(image_path))
        raise FileNotFoundError(f"No usable image is available for {image_path!r}")

    def list_dim(self, lst):
        if not isinstance(lst, list):
            return 0
        if not lst:
            return 1  # 空列表，当作一维
        return 1 + self.list_dim(lst[0])

    def forward(self, bounding_box=None, image_path: str = "", **kwargs) -> list:
        if self.debug:
            return "./cache/init_crop.png"

        # Keep compatibility with trajectories produced with the old public
        # argument name while exposing one canonical name to new rollouts.
        if bounding_box is None:
            bounding_box = kwargs.get("bound_boxes")
        
        try:
            # image_path = os.path.join(self.image_root, image_path)
            image_path = self._resolve_image_path(image_path)
            self.last_resolved_image_paths = [image_path]
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

        d_list = self.list_dim(bounding_box)
        if d_list == 1:
            bounding_box = [bounding_box]
            
        if not isinstance(bounding_box, list) or not all(len(box) == 4 for box in bounding_box):
            raise ValueError("Bounding boxes must be a list of [x1, y1, x2, y2]")

        # 裁剪图像并保存
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        folder = os.path.dirname(image_path)
        # output_dir = f"{base_name}_crops"
        # os.makedirs(output_dir, exist_ok=True)

        output_paths = []
        for idx, bbox in enumerate(bounding_box):
            cropped = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            output_path = os.path.join(folder, f"{base_name}_crop_obj_{idx}.jpg")
            cropped.save(output_path)
            if self.navigation_tool is not None:
                self.navigation_tool.register_derived_image(output_path, image_path)
            output_paths.append(output_path)

        return output_paths
