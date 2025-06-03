from transformers import Tool

class Detect3DTool(Tool):
    name = "detect3d"
    description = "3d detector"
    inputs = {
        "image_path": {
            "description": "The path to the image",
            "type": "string",
        },
    }
    output_type = "number"