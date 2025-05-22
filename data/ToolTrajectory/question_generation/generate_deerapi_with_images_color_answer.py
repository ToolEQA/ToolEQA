import os
from openai import OpenAI
import base64
import csv
from tqdm import tqdm
import http.client
import json
import pandas as pd
import re
import argparse
import pickle
import http.client
import time
import socket

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

def requests_api(images, prompt, max_retries=3, retry_delay=1):
    image_urls = []
    for image in images:
        base64_image = convert_image_to_base64(image)
        image_urls.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
    prompt = [{"type": "text", "text": prompt}]
    content = prompt + image_urls

    headers = {
        'Authorization': 'sk-lrenmYBYEOQH0rqv9rlMmoTaELkvZni1afswhr6be3tTN44S',
        'Content-Type': 'application/json'
    }
    
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 400
    })

    for attempt in range(max_retries):
        try:
            conn = http.client.HTTPSConnection('api.deerapi.com')
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            
            if res.status == 200:
                try:
                    data = json.loads(res.read().decode("utf-8"))
                    return data
                except json.JSONDecodeError:
                    print("Failed to decode JSON response")
            else:
                print(f"Request failed with status code: {res.status}")
                
        except Exception as e:
            print(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            
        finally:
            if 'conn' in locals():
                conn.close()
                
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            
    print("Max retries reached, giving up")
    return None

def post_process(result):
    result = result.strip("`\n")
    split = result.split("\n")
    print(split)

    # question = split[0][10:]
    # options = split[0][9:].strip("[]").split("; ")
    # options = [option[2:].strip() for option in options]
    answer = split[0][8:].strip("[]")

    # "scene", "floor", "question", "choices", "question_formatted", "answer", "label"
    response = {
        # "floor": 0,
        # "question": question,
        # "choices": options,
        # "question_formatted": f"{question} A) {options[0]} B) {options[1]} C) {options[2]} D) {options[3]}. Answer:",
        "answer": answer,
    }
    return response

def save_csv(csv_file_path, csv_columns, generated_data):
    with open(csv_file_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in generated_data:
            writer.writerow(data)



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


def extract_numbers_from_excel(question):
    
    pattern = re.compile(r'\((\d+)\)')
    numbers = set()

    # for question in df['question']:
    if isinstance(question, str):  # 只处理字符串
        matches = pattern.findall(question)
        numbers.update(matches)

    return numbers


def find_images_with_numbers(root_folder, numbers):
    from collections import defaultdict

    matched_images = defaultdict(list)
    numbers = set(numbers)

    # 遍历所有文件
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            for number in numbers:
                if file.startswith(f"{number}_"):
                    matched_images[number].append(os.path.join(subdir, file))
                    break

    # 检查是否每个数字都至少有一张图片
    if all(len(matched_images[num]) > 0 for num in numbers):
        # 合并所有图片路径为一个列表返回
        all_images = []
        for imgs in matched_images.values():
            all_images.extend(imgs)
        return all_images
    else:
        return []  # 有任意一个数字没有图片就返回空列表


def extract_numbers_from_locations(text):
# 正则表达式匹配括号中的数字
    # pattern = r"rug\s*\((\d+)\)"
    pattern = re.compile(r'\((\d+)\)')

    matches = re.findall(pattern, text)
    numbers = set()
    numbers.update(matches)
    # print('text', text, numbers)


    return numbers

    # indices = [int(num) for num in matches]

    # print(indices)


# def find_images_with_numbers(root_folder, numbers):
#     matched_images = []



#     for subdir, _, files in os.walk(root_folder):
#         for file in files:
#             for number in numbers:
#                 if file.startswith(f"{number}_"):
#                     matched_images.append(os.path.join(subdir, file))
#                     # break  # 一旦匹配就跳出 number 检查，加快速度

#     return matched_images

def generate(cfg):
    with open(cfg.prompt_path, "r", encoding="utf-8") as file:
        prompt = file.read()

    df = pd.read_csv(cfg.input_path)
    answers = []

    finished_scene = []
    if os.path.exists(cfg.output_path):
        out = pd.read_csv(cfg.output_path)
        finished_scene = out["scene_id"].tolist()

    for index, row in df.iterrows():
        scene_id = row['scene_id']
        if scene_id in finished_scene:
            print(f"skip scene {scene_id}")
            continue

        question = row['question']
        question_formatted = row['question_formatted']
        
        image_root = os.path.join(cfg.data_root,row['scene_id'],'objects_rgb')
        locations = row['locations']
        
        numbers = extract_numbers_from_locations(locations)
        matched_images = find_images_with_numbers(image_root, numbers)

        if len(matched_images) == 0:
            continue
        prompt_1 = prompt.format(question_formatted)
       
        result = requests_api(matched_images, prompt_1)
        if result is None:
            continue

        try:
            result_dict = post_process(result["choices"][0]["message"]["content"])
            result_dict["source_image"] = matched_images

            answers.append(result_dict["answer"])

            new_data = dict(row)
            new_data["answers"] = result_dict["answer"]

            write_csv(cfg.output_path, [new_data])

        except:
            continue


if __name__ == "__main__":
    # scene_root = "data/ReactEQA/sample_scene/images"

    parser = argparse.ArgumentParser()
    parser.add_argument("-prompt", "--prompt_path", help="prompt file path.", type=str, default="prompts_comparative_color_qa_2.txt")
    parser.add_argument("-type", "--question_type", help="the type of the question.", type=str)
    parser.add_argument("-root", "--data_root", help="the root dir of scene data.", type=str, default=r"C:\Users\Xiaomeng Fan\Desktop\期刊成稿\具身问答\format_withml\scene")
    parser.add_argument("-input", "--input_path", help="the path of output csv file.", type=str, default="question_HkseAnWCgqk_color_qa_3.csv")
    parser.add_argument("-output", "--output_path", help="the path of output csv file.", type=str, default="question_HkseAnWCgqk_color_qa_3.csv")
    parser.add_argument("-scene_number", help="generated questions numbers for scene.", type=int, default=10)

    args = parser.parse_args()
    print(args)

    generate(args)
