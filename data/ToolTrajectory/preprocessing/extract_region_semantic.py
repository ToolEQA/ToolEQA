# 提取每个区域的语义信息
import os
from generator_deerapi import requests_api
import json
from tqdm import tqdm
import random
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
from tqdm import tqdm

def sample_region_objects(root, region_name, sample_num):
    path = os.path.join(root, region_name)
    objs = os.listdir(path)

    samples = random.sample(objs, min(sample_num, len(objs)))

    images = []
    for sample in samples:
        # image = Image.open(os.path.join(path, sample))
        # images.append(image)
        images.append(os.path.join(path, sample))

    return images


def extract_region_semantic(root, model, processor, device):
    scenes_dir = os.listdir(root)
    scenes_dir.sort()

    prompt = "Analyze the given set of images (all captured from the same area within an indoor scene) and determine the most likely functional area they belong to. Respond ONLY with a single descriptive noun phrase, without explanations or additional text."
    region_semantic = []
    for scene_dir in tqdm(scenes_dir):
        if os.path.isdir(os.path.join(root, scene_dir)):
            regions_root = os.path.join(root, scene_dir, "objects_rgb")
            if os.path.isdir(regions_root):
                regions_dir = os.listdir(regions_root)
                for region_dir in tqdm(regions_dir):
                    images = sample_region_objects(regions_root, region_dir, 5)

                    response = get_response(model, processor, images, prompt, device)[0]

                    region_semantic.append({"region_id": region_dir, "region_name": response})

                scene_id = scene_dir.split("-")[-1]
                with open(os.path.join(root, scene_dir, f"{scene_id}.region.json"), "w") as f:
                    json.dump(region_semantic, f, indent=4)


def get_response(model, processor, images, prompt, device):
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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


if __name__=="__main__":
    root = "data/HM3D"

    device = "cuda"

    # Load tokenizer and model
    model_name = "/mynvme0/models/Qwen2-VL/Qwen2-VL-72B-Instruct-GPTQ-Int4/"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            device_map=device, 
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        ).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    extract_region_semantic(root, model, processor, device)
