from transformers import Tool

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


    def forward(self, object: str, image_path: str) -> list:
        pass