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


    # from src.llm_engine.qwen import QwenEngine
    # client = QwenEngine("/mynvme0/models/Qwen/Qwen2.5-VL-3B-Instruct")

    def forward_qwen(self, question, image_path: str) -> str:
        add_note = False
        if type(question) is not str:
            raise Exception("parameter question should be a string.")
        if not question:
            add_note = True
            question = "Please write a detailed caption for this image."
        
        image_paths = []
        if isinstance(image_path, str):
            image_paths = [image_path]
        else:
            print ('The type of input image is ', type(image_path))
            raise Exception("The type of input image should be string (image path)")

        messages = [
            {"role": "user", "content": question}
        ]
        output = self.client.call_vlm(
            messages,
            image_paths = image_paths
        )

        if add_note:
            output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

        return output

    def forward(self, question, prompt):
        return self.forward_qwen(question, prompt)
