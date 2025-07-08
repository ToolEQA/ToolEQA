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

    def forward(self, object: str, image_path: str) -> list:
        pass