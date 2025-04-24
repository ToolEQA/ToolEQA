import json
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class Qwen2vl():
    def __init__(self, model_name, device="cuda"):
        # Load tokenizer and model
        self.model_name = model_name
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name, 
                device_map=self.device, 
                torch_dtype="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True
            ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def get_response(self, images, prompt):
        # 创建对话信息
        message = {
                "role": "user",
                "content": [],
        }
        for image in images:
            message["content"].append({
                "type": "image",
                "image": image,
            })
        message["content"].append({
            "type": "text",
            "text": prompt,
        })
        messages = [message]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
