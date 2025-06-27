from transformers import Tool

class VQATool(Tool):
    name = "vqa"
    description = "A tool that can answer questions about attached images."
    inputs = {
        "question": {"description": "the question to answer", "type": "string"},
        "image_path": {
            "description": "The path to the image on which to answer the question",
            "type": "string",
        },
    }
    output_type = "string"

    def get_response(sefl, images, prompt):
        pass