# 不需要输入图片的问题生成

import os
from openai import OpenAI
import base64
import csv
from tqdm import tqdm
import http.client
import json

import random
import socket

import argparse
import pickle
import pandas as pd
import time

def load_data(path):
    postfix = path.split(".")[-1]
    with open(path, "r") as f:
        if postfix == "json":
            data = json.load(f)
        elif postfix == "pkl":
            data = pickle.load(f)
    return data


def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def requests_api(txt_files, prompt, max_retries=10, retry_delay=3):
    prompt_content = [{"type": "text", "text": prompt + "\n\n" + json.dumps(txt_files, indent=2)}]

    payload = json.dumps({
        "model": "gpt-4o-mini",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt_content
            }
            ],
            "max_tokens": 8192
        })
    headers = {
        'Authorization': 'sk-lrenmYBYEOQH0rqv9rlMmoTaELkvZni1afswhr6be3tTN44S',
        'Content-Type': 'application/json'
    }

    for attempt in range(max_retries):
        try:
            conn = http.client.HTTPSConnection('api.deerapi.com', timeout=10)
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            response_text = res.read().decode("utf-8")

            print(f"=== Attempt {attempt + 1} Response ===")
            print(repr(response_text))
            print("======================")

            if not response_text.strip():
                print("响应内容是空的！")
                raise ValueError("Empty response")
                
            data = json.loads(response_text)
            return data
        
        except Exception as e:
            print(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"{retry_delay}秒后重试...")
                time.sleep(retry_delay)
            else:
                print("已达到最大重试次数，请求失败")
                raise Exception(f"API请求失败: {str(e)}")
        finally:
            conn.close()  # 确保连接被关闭


def post_process_multiple(result):
    
    result = result.strip("`\n")

    # Use the '---' separator to split each question block
    question_blocks = [block.strip() for block in result.split('---') if block.strip()]
    parsed_questions = []

    for block in question_blocks:
        lines = block.split("\n")

        try:
            question = lines[0][10:].strip()
            options = lines[1][9:].strip("[]").split("; ")
            options = [opt[2:].strip() for opt in options]  # remove 'A. ', 'B. ', etc.
            answer = lines[2][8:].strip()
            locations = lines[3][16:].strip()

            parsed_questions.append({
                "question": question,
                "choices": options,
                "question_formatted": f"{question} A) {options[0]} B) {options[1]} C) {options[2]} D) {options[3]}. Answer:",
                "answer": answer,
                "locations": locations
            })

        except Exception as e:
            print(f"Error parsing block:\n{block}\nException: {e}")

    return parsed_questions


def post_process(result):
    result = result.strip("`\n")
    split = result.split("\n")
    print(split)

    question = split[0][10:]
    options = split[1][9:].strip("[]").split("; ")
    options = [option[2:].strip() for option in options]
    answer = split[2][8:].strip("[]")[0]
    locations = split[3][16:].strip("[]")

    # "scene", "floor", "question", "choices", "question_formatted", "answer", "label"
    response = {
        # "floor": 0,
        "question": question,
        "choices": options,
        "question_formatted": f"{question} A) {options[0]} B) {options[1]} C) {options[2]} D) {options[3]}. Answer:",
        "answer": answer,
        "locations": locations
    }
    return response

def save_csv(csv_file_path, csv_columns, generated_data):
    with open(csv_file_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in generated_data:
            writer.writerow(data)


def json_generator(cfg, scene_id, objects_data, prompt, floor_name, generate_number):
    results = []
    for num in range(generate_number):
        random.shuffle(objects_data)
        result = requests_api(objects_data, prompt)
            # result_dict = post_process(result["choices"][0]["message"]["content"])
        result_dict_list = post_process_multiple(result["choices"][0]["message"]["content"])
        # print('result', result)
        # print('len', len(result_dict_list))
        for idx, q in enumerate(result_dict_list):
            print(f"Question {idx + 1}:")
            print(q["question_formatted"])
            print(f"Answer: {q['answer']} \n")
        # result_dict["source_image"] = file
        # result_dict["scene"] = file.split("_")[0]
        for i in range(len (result_dict_list)):
            result = result_dict_list[i]
            result["scene_id"] = scene_id
            result["label"] = cfg.question_type
            result["floor_id"] = floor_name

            results.append(result_dict_list[i])

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
    floor_number = [-1,1,2,3]

    finished_scene = []
    if os.path.exists(cfg.output_path):
        df = pd.read_csv(cfg.output_path)
        finished_scene = df["scene_id"].tolist()

    for scene_data in all_scenes_path:
        if scene_data in finished_scene:
            print(f"skip scene {scene_data}")
            continue
        if not os.path.isdir(os.path.join(cfg.data_root, scene_data)):
            continue
        scene_id = scene_data.split("-")[-1]
        print('scene_id', scene_id)
        # objects_data_path = os.path.join(cfg.data_root, scene_data, scene_id + ".objects")


        for f_n in floor_number:
            objects_data_path = os.path.join(cfg.data_root, scene_data, scene_id + ".objects.cleaned.floor" + str(f_n))
            if os.path.exists(objects_data_path + ".json"):
                objects_data_path = objects_data_path + ".json"
            else: 
                continue

            print('objects_data_path', objects_data_path)

            objects_data = load_data(objects_data_path)
            new_objects_data = []
            # 过滤类别
            for obj_info in objects_data:
                if obj_info["category_name"] in categories:
                    new_objects_data.append(obj_info)
            generate_number = int(len(new_objects_data)*cfg.scene_number)
            print('generate_number', generate_number)

            results = json_generator(cfg, scene_data, new_objects_data, prompt, f_n, generate_number)
            write_csv(cfg.output_path, results)

    #     print('objects_data_path', objects_data_path)
    #     if os.path.exists(objects_data_path + ".pkl"):
    #         objects_data_path = objects_data_path + ".pkl"
    #     elif os.path.exists(objects_data_path + ".json"):
    #         objects_data_path = objects_data_path + ".json"

    #     objects_data = load_data(objects_data_path)

    #     results = json_generator(cfg, scene_data, objects_data, prompt)

    #     break
    # write_csv(cfg.output_path, results)




    # with open('HkseAnWCgqk_objects.json', 'r', encoding='utf-8') as f:
    #     lines = json.load(f)

    
    
    # for i in range(10):
    #     result = requests_api(lines, prompt)
    #     # result_dict = post_process(result["choices"][0]["message"]["content"])
    #     result_dict_list = post_process_multiple(result["choices"][0]["message"]["content"])
    #     print('result', result)
    #     print('len', len(result_dict_list))
    #     for idx, q in enumerate(result_dict_list):
    #         print(f"Question {idx + 1}:")
    #         print(q["question_formatted"])
    #         print(f"Answer: {q['answer']} ({q['label']})\n")
    #     # result_dict["source_image"] = file
    #     # result_dict["scene"] = file.split("_")[0]
    #     for i in range(len (result_dict_list)):
    #         generated_data.append(result_dict_list[i])
        

        # try:
        #     result = requests_api(image_file, prompt)
        #     result_dict = post_process(result["choices"][0]["message"]["content"])
        #     result_dict["source_image"] = file
        #     result_dict["scene"] = file.split("_")[0]
        #     generated_data.append(result_dict)
        # except:
        #     continue
        # for i in range(len (result_dict_list)):
        #     writer.writerow(result_dict_list[i])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-prompt", "--prompt_path", help="prompt file path.", type=str, default="prompts/attribute/prompts_comparative_size.txt")
    parser.add_argument("-type", "--question_type", help="the type of the question.", type=str, default='attribute')
    parser.add_argument("-root", "--data_root", help="the root dir of scene data.", type=str, default=r"C:\Users\Xiaomeng Fan\Desktop\期刊成稿\具身问答\format_withml\scene")
    parser.add_argument("-output", "--output_path", help="the path of output csv file.", type=str, default="question_HkseAnWCgqk.csv")
    parser.add_argument("-scene_number", help="generated questions numbers for scene.", type=int, default=0.1)

    args = parser.parse_args()
    print(args)

    generate(args)
