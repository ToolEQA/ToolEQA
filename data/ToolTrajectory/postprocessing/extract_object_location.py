from data.ToolTrajectory.preprocessing.vlm_request import Qwen2vl
import os
import json
from PIL import Image
from data.ToolTrajectory.generator_deerapi import requests_api

def get_obj_location_from_id(model, 
                             scene_id: str, 
                             object_id: int, 
                             prompt="Provide a detailed spatial description of the [{}], using references to nearby objects or layout. The object is visible in multiple images from different viewpoints. Summarize into a phrase, for example: TV on the cabinet next to the window. Only answer the phrase.", 
                             root="data/HM3D"):
    # search image files
    region_images_root = os.path.join(root, scene_id, "objects_rgb")
    target_images_path = []
    if os.path.isdir(region_images_root):
        index = 0
        for roots, dirs, files in os.walk(region_images_root):
            if index == 0:
                index += 1
                continue

            for file in files:
                obj_id = int(file.split("_")[0])
                if obj_id == object_id:
                    target_images_path.append(os.path.join(roots, file))
    if len(target_images_path) == 0:
        print("没有检索到图像")
        return None
    
    obj_type = target_images_path[0].split("/")[-1].split("_")[1].replace("-", " ")
    prompt = prompt.format(obj_type)
    
    if model is not None:
        response = model.get_response(target_images_path, prompt)
    else:
        response = requests_api(target_images_path, prompt)
        response = response["choices"][0]["message"]["content"]
    return response

if __name__=="__main__":
    model_name = "/mynvme0/models/Qwen2-VL/Qwen2-VL-72B-Instruct-GPTQ-Int4/"
    model = Qwen2vl(model_name)
    prompt = "Provide a detailed spatial description of the [{}], using references to nearby objects or layout. The object is visible in multiple images from different viewpoints. Summarize into a phrase, for example: TV on the cabinet next to the window. Only answer the phrase."
    response = get_obj_location_from_id(model, "00006-HkseAnWCgqk", 14, prompt)
    print(response)