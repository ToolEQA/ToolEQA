from transformers import Tool

class GoNextPointTool(Tool):
    name = "go_next_point"
    description = "the agent conitnue explore next point."
    inputs = {
        "image_path": {
            "description": "The path to the image",
            "type": "string",
        },
    }
    output_type = "any"