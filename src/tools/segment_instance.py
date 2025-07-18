from transformers import Tool
import cv2
import numpy as np
import base64
import requests
from PIL import Image
from src.utils.shared_memory import client_send_image

class SegmentInstanceTool(Tool):
    name = "SegmentInstanceTool"
    description = "A tool that can do instance segmentation on the given image."
    inputs = {
        "image_path": {
            "description": "The path of image that the tool can read.",
            "type": "string",
        },
        "prompt": {
            "description": "The bounding box that you want this model to segment. The bounding boxes could be from user input or tool `objectlocation`. You can set it as None or empty list to enable 'Segment Anything' mode.",
            "type": "any",
        },
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = kwargs.get("debug", False)
        if self.debug:
            return
        
        self.endpoint = "location_seg"

    def forward(self, image_path: str, prompt: str) -> str:
        if self.debug:
            return np.zeros((640,640))
        
        image = np.array(Image.open(image_path).convert("RGB"))

        data = {
            'endpoint': self.endpoint,
            'image': image,
            'text': prompt
        }
        res = client_send_image(data)

        if "error" in res.keys():
            raise Exception(f"Error: {res['error']}")
        return res

    def draw_seg(img, masks, output="output_seg.jpg"):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for mask in masks:
            mask = np.array(mask, dtype=np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(img_bgr, [contour], -1, (0, 255, 0), 3)  # 绿色轮廓
        cv2.imwrite(output, img_bgr)


if __name__=="__main__":
    tool = SegmentInstanceTool()
    import time
    t = time.time()
    result = tool.forward("./tmp/human.jpg", "human")
    print("Time taken:", time.time() - t)
    print(result.keys())