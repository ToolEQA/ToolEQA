from transformers import Tool

class Detect2DTool(Tool):
    name = "detect2d"
    description = "2D detector"
    inputs = {
        "image_path": {
            "description": "The path to the image",
            "type": "string",
        },
    }
    output_type = "number"

    def detect():
        pass