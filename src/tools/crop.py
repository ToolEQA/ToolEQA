from transformers import Tool

class ObjectCrop(Tool):
    name = "ObjectCrop"
    description = "Given the bounding boxes of objects, crop and save the relevant objects from the image."
    inputs = {
        "bound_boxes": {"description": "the bounding boxes of objects", "type": "string"},
        "image_path": {
            "description": "The path to the image on which to crop objects.",
            "type": "string",
        },
    }
    output_type = "string"


    def forward(self, object: str, image_path: str) -> list:
        pass