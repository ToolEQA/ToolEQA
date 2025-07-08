from transformers import Tool

class SegmentInstanceTool(Tool):
    name = "SegmentInstanceTool"
    description = "A tool that can do instance segmentation on the given image."
    inputs = {
        "image_path": {
            "description": "The path of image that the tool can read.",
            "type": "string",
        },
        "prompt": {
            "description": "The bounding box that you want this model to segment. The bounding boxes could be from user input or tool `objectlocation`. You can set it as None or empty list to enable 'Segment Anything' mode.",
            "type": "any",
        },
    }
    output_type = "string"

    def forward(self, image_path: str, prompt: str) -> str:
        pass