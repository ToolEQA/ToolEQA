# 将生成的问题中的obj转换为语言位置描述（lamp(17) -> lamp on the bedside table next to the bed）
from data.ToolTrajectory.preprocessing.vlm_request import Qwen2vl
import os
import json
from PIL import Image
from data.ToolTrajectory.generator_deerapi import requests_api
import re
import csv

def extract_obj_id(question):
    
    pattern = re.compile(r'\((\d+)\)')
    numbers = set()

    # for question in df['question']:
    if isinstance(question, str):  # 只处理字符串
        matches = pattern.findall(question)
        numbers.update(matches)

    return numbers


def get_image_from_id(scene_id: str, object_id: int, root="data/HM3D"):
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
        # print("没有检索到图像")
        return None
    return target_images_path


def get_obj_desc_from_id(model, 
                             scene_id: str, 
                             object_id: int, 
                             prompt="Provide a detailed spatial description of the [{}], using references to nearby objects or layout. The object is visible in multiple images from different viewpoints. Summarize into a phrase, for example: TV on the cabinet next to the window. Only answer the phrase.", 
                             root="data/HM3D"):
    target_images_path = get_image_from_id(scene_id, object_id)
    
    obj_type = target_images_path[0].split("/")[-1].split("_")[1].replace("-", " ")
    prompt = prompt.format(obj_type)
    
    if model is not None:
        response = model.get_response(target_images_path, prompt)
    else:
        response = requests_api(target_images_path, prompt)
        response = response["choices"][0]["message"]["content"]
    return response

def transfer_question(scene_id, question, locations):
    
    objs_name = [name.split(":")[0] for name in locations.split("; ")]
    objs_id = [int(name.split(" (")[1].strip("()")) for name in objs_name]
    
    if len(objs_name) == 0:
        return question
    
    objs_desc = []
    for id in objs_id:
        res = get_obj_desc_from_id(None, scene_id, int(id))
        if res is None:
            break
        objs_desc.append(res.strip(".").lower())
    
    if len(objs_desc) < len(objs_name):
        return question

    for obj_name, obj_desc in zip(objs_name, objs_desc):
        print(obj_name, "====>", obj_desc)
        question = question.replace(obj_name, obj_desc)
    
    return question

if __name__=="__main__":
    csv_file_path = "tmp/question_HkseAnWCgqk_color_onlyans_2.csv"
    output_data = []
    title = None
    
    with open(csv_file_path, "r") as f:
        csv_reader = csv.DictReader(f, skipinitialspace=True)
        title = csv_reader.keys()
        for idx, row in enumerate(csv_reader):
            scene_id = row["scene_id"]
            question = row["question"]
            locations = row["locations"]
            new_question = transfer_question(scene_id, question, locations)
            
            row["question"] = new_question
            output_data.append(row)

    with open("tmp/output.csv", "w", newline="", encoding="utf-8") as csvfile:
        # 获取字典的键作为CSV表头
        writer = csv.DictWriter(csvfile, fieldnames=title)
        
        writer.writeheader()  # 写入表头
        writer.writerows(output_data)  # 写入所有行