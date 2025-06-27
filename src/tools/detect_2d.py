from transformers import Tool

# authorized_types = ["string", "integer", "number", "image", "audio", "any", "boolean"]
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

    def forward(self):
        pass