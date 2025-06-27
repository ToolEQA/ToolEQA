import json
import os
import csv
import random
import uuid
import base64
import re
import subprocess
from data.ToolTrajectory.postprocessing.extract_object_location import get_image_from_id

def short_uuid():
    u = uuid.uuid4()
    return base64.urlsafe_b64encode(u.bytes).rstrip(b'=').decode('ascii')

def read_csv(csv_file_path):
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        data_list = list(csv.DictReader(file))
    return data_list

def objstr2list(objs_str):
    pattern = r'^(?P<name>.+?)\s+\((?P<id>\d+)\):\s+\[(?P<position>[-\d.,\s]+)\]'
    objs_list = objs_str.split("; ")
    objs_data = []
    for obj in objs_list:
        match = re.match(pattern, obj.strip())
        if match:
            name = match.group('name').strip()
            id = match.group('id')
            pos = [float(p) for p in match.group('position').split(", ")]
            if len(pos) != 3:
                return None
            objs_data.append({
                "name": name,
                "id": int(id),
                "pos": [pos[0], pos[2], -pos[1]],
            })

    if len(objs_list) == len(objs_data):
        return objs_data
    return None

if __name__ == "__main__":
    scene_root = "data/HM3D"
    question_files = [
        "data/ToolTrajectory/questions/final_question/attribute/color.csv",
        "data/ToolTrajectory/questions/final_question/attribute/size.csv",
        "data/ToolTrajectory/questions/final_question/attribute/special.csv",
        "data/ToolTrajectory/questions/final_question/counting/counting.csv",
        "data/ToolTrajectory/questions/final_question/distance/distance.csv",
        "data/ToolTrajectory/questions/final_question/location/location.csv",
        "data/ToolTrajectory/questions/final_question/location/special.csv",
        "data/ToolTrajectory/questions/final_question/relationship/relationship.csv",
        "data/ToolTrajectory/questions/final_question/status/status.csv"
    ]

    samples_data = []
    for question_file in question_files:
        questions = random.sample(read_csv(question_file), int(2000 / 9))
        print(f"Processing {question_file} with {len(questions)} questions.")

        question_cate = question_file.split("/")[-2]
        question_sub_cat = question_file.split("/")[-1].split(".")[0]

        for question in questions:
            sample_data = {}

            question_id = short_uuid()
            scene_id = question["scene_id"]
            objs_data = objstr2list(question["locations"])
            if objs_data is None:
                continue
            for idx, obj in enumerate(objs_data):
                images_path = get_image_from_id(scene_id, obj["id"])
                if images_path is None:
                    continue
                obj['image'] = []
                for j, image_path in enumerate(images_path):
                    save_image_root = f"/mynvme1/ReactEQA-Samples/{question_id}"
                    if not os.path.exists(save_image_root):
                        os.makedirs(save_image_root, exist_ok=True)
                    save_image_path = os.path.join(save_image_root, f"{obj['id']}_{idx}-{j}.jpg")
                    subprocess.run(["cp", image_path, save_image_path], check=True)
                    obj['image'].append(save_image_path)

            sample_data["sample_id"] = question_id
            sample_data["scene"] = scene_id
            sample_data["question"] = question["question"]
            sample_data["proposals"] = eval(question["choices"])
            sample_data["answer"] = question["answer"]
            sample_data["question_type"] = f"{question_cate}-{question_sub_cat}"
            sample_data["related_objects"] = objs_data
            samples_data.append(sample_data)
        print(f"Processed {len(samples_data)} samples so far.")

    save_path = "/mynvme1/ReactEQA-Samples/sample_question.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(samples_data, f, indent=4, ensure_ascii=False)