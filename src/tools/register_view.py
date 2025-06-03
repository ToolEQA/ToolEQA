from transformers import Tool

class RigisterViewTool(Tool):
    name = "register_view"
    description = "record images related to the user question"
    inputs = {
        "image_path": {
            "description": "The path to the image",
            "type": "string",
        },
    }
    output_type = "any"