import cv2
import base64
import requests
import numpy as np
import os
from src.utils.shared_memory import client_send_image
from PIL import Image
from src.tools.compat import Tool
from omegaconf import OmegaConf

# authorized_types = ["string", "integer", "number", "image", "audio", "any", "boolean"]
class ObjectLocation2D(Tool):
    name = "ObjectLocation2D"
    description = (
        "Localize objects in an image and return their 2D bounding boxes. Use the exact image "
        "path returned by GoNextPointTool."
    )
    inputs = {
        "object": {"description": "the object that need to be localized", "type": "string"},
        "image_path": {
            "description": "The path to the image on which to localize objects.",
            "type": "string",
        },
    }
    output_type = "any"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = kwargs.get("debug", False)
        self.gpu_id = kwargs.get("gpu_id", 0)
        self.args = kwargs.get("args", None)
        self.last_resolved_image_paths = []
        self.navigation_tool = None
        if self.debug:
            return
        
        self.cfg = OmegaConf.load(self.args.cfg)
        OmegaConf.resolve(self.cfg)

        self.endpoint = "location_2d"

    def bind_navigation_tool(self, navigation_tool):
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
            return [0, 0, 0, 0]
        
        image_path = self._resolve_image_path(image_path)
        image = np.array(Image.open(image_path).convert("RGB"))
        self.last_resolved_image_paths = [os.path.realpath(os.path.abspath(image_path))]

        data = {
            'endpoint': self.endpoint,
            'image': image,
            'text': object
        }

        res = client_send_image(data, self.gpu_id)
        if "error" in res.keys():
            error = str(res["error"])
            if "no object" in error.lower() or "not found" in error.lower():
                return {
                    "bboxes_2d": [],
                    "labels": [],
                    "scores": [],
                    "image_size": [int(image.shape[1]), int(image.shape[0])],
                }
            raise RuntimeError(f"2D localization failed for {object!r}: {error}")
        res["image_size"] = [int(image.shape[1]), int(image.shape[0])]
        return res

    def draw_2dbox(self, img, bboxes, labels=None, output="output_det2d.jpg"):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            top_left = (x1, y1)
            bottom_right = (x2, y2)
            color = (0, 0, 255)  # 红色 (B, G, R)
            thickness = 2
            cv2.rectangle(img_bgr, top_left, bottom_right, color, thickness)
        cv2.imwrite(output, img_bgr)

if __name__=="__main__":
    tool = ObjectLocation2D()
    import time
    t = time.time()
    result = tool.forward("light", "tmp/1-4.png")
    print("Time taken:", time.time() - t)
    print(result)
