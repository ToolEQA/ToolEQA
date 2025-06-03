from transformers import Tool

class QueryObjectInfoTool(Tool):
    name = "query_object_info"
    description = "query the information of object from semantic map."
    inputs = {
        "image_path": {
            "description": "The path to the image",
            "type": "string",
        },
    }
    output_type = "any"