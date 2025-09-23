from transformers import Tool
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

class ObjectLocation3D(Tool):
    name = "ObjectLocation3D"
    description = "Localize 3D objects in the scene and return their 3D bounding boxes and center coordinates."
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
        if self.debug:
            return
        
        self.cfg = OmegaConf.load(self.args.cfg)
        OmegaConf.resolve(self.cfg)

        # Initialize any necessary components for 3D object localization here
        # For example, you might load a pre-trained model or set up a 3D environment
        self.endpoint = "location_3d"

    def forward(self, object: str, image_path: str) -> list:
        if self.debug:
            return [0,0,0], [0,0,0], [[1,0,0],[0,1,0],[0,0,1]]
        
        if not os.path.exists(image_path):
            if not os.path.exists("/"+image_path):
                image_path = os.path.join(self.cfg.output_dir, image_path)
            else:
                image_path = "/" + image_path
        image = np.array(Image.open(image_path).convert("RGB"))

        data = {
            'endpoint': self.endpoint,
            'image': image,
            'text': object
        }
        res = client_send_image(data, self.gpu_id - 4)

        if "error" in res.keys():
            print(f"Error: {object}, {image_path}, {res['error']}")
            return None, None, None
            # raise Exception(f"Error: {object}, {image_path}, {res['error']}")

        center = [[round(bbox[0], 2), round(bbox[1], 2), round(bbox[2], 2)] for bbox in res["bboxes_3d"].cpu().tolist()]
        size = [[round(bbox[3], 2), round(bbox[4], 2), round(bbox[5], 2)] for bbox in res["bboxes_3d"].cpu().tolist()]
        # yaw = [bbox[6] for bbox in res["bboxes_3d"]]
        rot = res["rot_mat"]

        return center, size

if __name__=="__main__":
    tool = ObjectLocation3D()
    import time
    t = time.time()
    result = tool.forward("light", "/mynvme1/EQA-Traj-0720/Z_HvYhe6T6e3msJ7HRsPAg/1-4.png")
    print("Time taken:", time.time() - t)
    print(np.array(result['rot_mat'][0]).shape)