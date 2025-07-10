from transformers import Tool
import requests
import base64
from PIL import Image
import io

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
        self.url = "http://localhost:8000/location_3d"


    def forward(self, object: str, image_path: str) -> list:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        data = {
            'image': image_b64,
            'text': object
        }

        res = requests.post(self.url, json=data)
        if res.status_code != 200:
            raise Exception(f"Error in ObjectLocation3D: {res.status_code} - {res.text}")
        return res.json()
    
if __name__=="__main__":
    tool = ObjectLocation3D()
    import time
    t = time.time()
    result = tool.forward("human", "./tmp/human.jpg")
    print("Time taken:", time.time() - t)
    print(result)