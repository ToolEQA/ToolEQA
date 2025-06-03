from transformers import Tool

class ObjectLocationTool(Tool):
    name = "object_location"
    description = "A tool that can localize objects in given images, outputing the bounding boxes of the objects."
    inputs = {
        "object": {"description": "the object that need to be localized", "type": "string"},
        "image_path": {
            "description": "The path to the image on which to localize objects. This should be a local path to downloaded image.",
            "type": "string",
        },
    }
    output_type = "any"