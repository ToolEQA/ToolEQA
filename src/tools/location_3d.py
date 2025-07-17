from transformers import Tool
import requests
import base64
from PIL import Image
import io
from src.utils.shared_memory import client_send_image
import numpy as np

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
        # Initialize any necessary components for 3D object localization here
        # For example, you might load a pre-trained model or set up a 3D environment
        self.endpoint = "location_3d"

    def forward(self, object: str, image_path: str) -> list:
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
    
if __name__=="__main__":
    tool = ObjectLocation3D()
    import time
    t = time.time()
    result = tool.forward("human", "./tmp/human.jpg")
    print("Time taken:", time.time() - t)
    print(np.array(result['rot_mat'][0]).shape)