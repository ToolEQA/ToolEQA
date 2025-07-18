import cv2
import base64
import requests
import numpy as np
from src.utils.shared_memory import client_send_image
from PIL import Image
from transformers import Tool

# authorized_types = ["string", "integer", "number", "image", "audio", "any", "boolean"]
class ObjectLocation2D(Tool):
    name = "ObjectLocation2D"
    description = "A tool that can localize objects in given images, outputing the bounding boxes of the objects."
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
        if self.debug:
            return

        self.endpoint = "location_2d"

    def forward(self, object: str, image_path: str) -> list:
        if self.debug:
            return [0, 0, 0, 0]
        
        image = np.array(Image.open(image_path).convert("RGB"))

        data = {
            'endpoint': self.endpoint,
            'image': image,
            'text': object
        }

        res = client_send_image(data)
        
        if "error" in res.keys():
            raise Exception(f"Error: {res['error']}")
        
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
    result = tool.forward("human", "./tmp/human.jpg")
    print("Time taken:", time.time() - t)
    print(result.keys())