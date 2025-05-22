import os
import json
import pickle
import argparse
import random
import csv
import pandas as pd

from tqdm import tqdm
from data.ToolTrajectory.generator_deerapi import requests_api
from data.ToolTrajectory.postprocessing.replace_objs_locations import get_image_from_id

def load_data(path):
    postfix = path.split(".")[-1]
    if postfix == "json":
        with open(path, "r") as f:
            data = json.load(f)
        return data
    elif postfix == "pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        print("file error.")

def post_process(result, extra):
    print(result)
    result = result.strip("`\n")
    splits = result.split("\n")
    if len(splits) < 3:
        return None
    question = splits[0][10:]
    options = splits[1].split(":")[-1].strip(" []").split("; ")
    if len(options) == 4:
        options = [option[2:].strip() for option in options]
    else:
        return None
    answer = splits[2][8:].strip("[]")

    # "scene", "floor", "question", "choices", "question_formatted", "answer", "label"
    response = {
        # "floor": 0,
        "question": question,
        "choices": options,
        "question_formatted": f"{question} A) {options[0]} B) {options[1]} C) {options[2]} D) {options[3]}. Answer:",
        "answer": answer,
    }
    response.update(extra)
    return response

def post_process_multi(results, extra):
    results = results.strip("`\n")
    # print(results)
    split = results.split("---")
    responses = []
    for item in split:
        response = post_process(item, extra)
        if response is not None:
            responses.append(response)
    return responses

def single_obj_generator(cfg, scene_id, objects_data, prompt, number=20):
    results = []

    for obj in objects_data:
        obj_cate = obj["category_name"]
        obj_id = obj["object_id"]
        obj_center = obj["bbox"]["center"]
        obj_floor = obj["floor_id"]
        
        obj_images = get_image_from_id(scene_id, obj_id, cfg.data_root)
        if obj_images is None:
            continue
        new_prompt = prompt.format(obj_cate)
        response = requests_api(obj_images, new_prompt)
        if response is None:
            continue
        print("==========================")
        # print(response["choices"][0]["message"]["content"])
        
        extra_data = {}
        extra_data["scene_id"] = scene_id
        extra_data["label"] = cfg.question_type
        extra_data["locations"] = obj_cate + f" ({obj_id}): " + str([round(p, 3) for p in obj_center])
        extra_data["floor_id"] = obj_floor
        result = post_process_multi(response["choices"][0]["message"]["content"], extra_data)

        results.extend(result)
        if len(results) > number:
            break

    return results


def write_csv(file_path, data):
    if len(data) > 0:
        title = data[0].keys()
    else:
        return

    if os.path.exists(file_path):
        with open(file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=title)
            writer.writerows(data)
    else:
        with open(file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=title)
            writer.writeheader()
            writer.writerows(data)


def generate(cfg):
    with open(cfg.prompt_path, "r", encoding="utf-8") as file:
        prompt = file.read()

    categories = []
    with open(os.path.join(cfg.data_root, "new2ori_mapping.json"), "r") as f:
        categories_json = json.load(f)
        for big_cate in categories_json:
            categories.extend(categories_json[big_cate])

    all_scenes_path = os.listdir(cfg.data_root)
    all_scenes_path.sort()

    finished_scene = []
    if os.path.exists(cfg.output_path):
        df = pd.read_csv(cfg.output_path)
        finished_scene = df["scene_id"].tolist()

    for scene_data in all_scenes_path:
        if scene_data in finished_scene:
            print(f"skip scene {scene_data}")
            continue
        print(f"processing {scene_data}")
        if not os.path.isdir(os.path.join(cfg.data_root, scene_data)):
            continue
        scene_id = scene_data.split("-")[-1]
        objects_data_path = os.path.join(cfg.data_root, scene_data, scene_id + ".objects")
        if os.path.exists(objects_data_path + ".extra.pkl"):
            objects_data_path = objects_data_path + ".extra.pkl"
        elif os.path.exists(objects_data_path + ".json"):
            objects_data_path = objects_data_path + ".json"

        objects_data = load_data(objects_data_path)
        new_objects_data = []
        # 过滤类别
        for obj_info in objects_data["objects"]:
            if obj_info["category_name"] in categories:
                new_objects_data.append(obj_info)

        results = single_obj_generator(cfg, scene_data, new_objects_data, prompt)

        write_csv(cfg.output_path, results)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-prompt", "--prompt_path", help="prompt file path.", type=str)
    parser.add_argument("-type", "--question_type", help="the type of the question.", type=str)
    parser.add_argument("-root", "--data_root", help="the root dir of scene data.", type=str, default="./data/HM3D")
    parser.add_argument("-output", "--output_path", help="the path of output csv file.", type=str, default="./data/HM3D")
    args = parser.parse_args()
    print(args)

    generate(args)
