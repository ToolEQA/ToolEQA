import os
from openai import OpenAI
import base64
import csv
from tqdm import tqdm
import http.client
import json

import random

def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def requests_api(txt_files, prompt):
    # image_urls = []
    # for image in images:
    #     base64_image = convert_image_to_base64(image)
    #     image_urls.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
    # prompt = [{"type": "text", "text": prompt+ "\n\n" + txt_files}]
    # import json

    prompt = [{"type": "text", "text": prompt + "\n\n" + json.dumps(txt_files, indent=2)}]

    content = prompt 

    conn = http.client.HTTPSConnection('api.deerapi.com')
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": content
            }
            ],
            "max_tokens": 8192
        })
    headers = {
        'Authorization': 'sk-lrenmYBYEOQH0rqv9rlMmoTaELkvZni1afswhr6be3tTN44S',
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    response_text = res.read().decode("utf-8")  # 🔁 先读取并解码
    print("=== Response Text ===")
    print(repr(response_text))  # 用 repr 看有没有空格、换行或其他乱码
    print("======================")
    if response_text.strip():
        data = json.loads(response_text)        # ✅ 再解析 JSON
    else:
        print("响应内容是空的！")
    # if res.text.strip():
    #     data = json.loads(res.read().decode("utf-8"))
    # else:
    #     print("响应内容是空的！")
    

    return data


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
            label = lines[3][7:].strip()
            locations = lines[4][16:].strip()

            parsed_questions.append({
                "question": question,
                "choices": options,
                "question_formatted": f"{question} A) {options[0]} B) {options[1]} C) {options[2]} D) {options[3]}. Answer:",
                "answer": answer,
                "label": label,
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
    label = split[3][7:].strip("[]")
    locations = split[4][16:].strip("[]")

    # "scene", "floor", "question", "choices", "question_formatted", "answer", "label"
    response = {
        # "floor": 0,
        "question": question,
        "choices": options,
        "question_formatted": f"{question} A) {options[0]} B) {options[1]} C) {options[2]} D) {options[3]}. Answer:",
        "answer": answer,
        "label": label,
        "locations": locations
    }
    return response

def save_csv(csv_file_path, csv_columns, generated_data):
    with open(csv_file_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in generated_data:
            writer.writerow(data)

if __name__ == "__main__":
    # scene_root = "data/ReactEQA/sample_scene/images"
    prompt_file = "prompts_location_2.txt"

    # prompt_file = "prompts_counting_2.txt"

    csv_columns = [ "question", "choices", "question_formatted", "answer", "label", "locations"]

    generated_data = []
    index = 0
    # scene_dir = os.listdir(scene_root)
    # scene_dir.sort()
    
    csv_file_path = 'question_HkseAnWCgqk.csv'

    # finished_samples = []
    # if os.path.exists(csv_file_path):
    #     with open(csv_file_path, 'r') as csvfile:
    #         reader = csv.DictReader(csvfile)
    #         for row in reader:
    #             finished_samples.append(row['source_image'])

    csvfile = open(csv_file_path, "a", newline='')
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)

    with open(prompt_file, "r", encoding="utf-8") as file:
        prompt = file.read()

    # with open(prompt_file, "r") as file:
    #     prompt = file.read()

    # with open('output.txt', 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    # with open('HkseAnWCgqk_objects.json', 'r', encoding='utf-8') as f:
    #     lines =  f.read()

    with open('HkseAnWCgqk_objects.json', 'r', encoding='utf-8') as f:
        lines = json.load(f)

    random.shuffle(lines)
    # for scene in tqdm(scene_dir):
    #     if index > 100:
    #         break

        # index += 1
        # images_file = os.listdir(os.path.join(scene_root, scene))
        # images_file.sort()
        # images_file = [os.path.join(scene_root, scene, file) for file in images_file]
    for i in range(10):
        result = requests_api(lines, prompt)
        # result_dict = post_process(result["choices"][0]["message"]["content"])
        result_dict_list = post_process_multiple(result["choices"][0]["message"]["content"])
        print('result', result)
        print('len', len(result_dict_list))
        for idx, q in enumerate(result_dict_list):
            print(f"Question {idx + 1}:")
            print(q["question_formatted"])
            print(f"Answer: {q['answer']} ({q['label']})\n")
        # result_dict["source_image"] = file
        # result_dict["scene"] = file.split("_")[0]
        for i in range(len (result_dict_list)):
            generated_data.append(result_dict_list[i])
        

        # try:
        #     result = requests_api(image_file, prompt)
        #     result_dict = post_process(result["choices"][0]["message"]["content"])
        #     result_dict["source_image"] = file
        #     result_dict["scene"] = file.split("_")[0]
        #     generated_data.append(result_dict)
        # except:
        #     continue
        for i in range(len (result_dict_list)):
            writer.writerow(result_dict_list[i])