import os
import json
import numpy as np
import ast
import math
import jsonlines
import cv2
import base64
from collections import defaultdict
from tqdm import tqdm
from src.vlm.generator_deerapi import requests_api
from src.llm_engine.gpt import GPTEngine

def convert_image_to_base64(image):
    if os.path.exists(image):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image, np.ndarray):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    return None

def prepare_messages(images, prompt):
    if images is not None:
        image_urls = []
        if isinstance(images, (list, tuple)):
            # 是列表或元组
            images = images
        else:
            # 不是列表，那就当成一个单路径，封装成列表
            images = [images]

        for image in images:
            base64_image = convert_image_to_base64(image)
            image_urls.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
        prompt = [{"type": "text", "text": prompt}]
        content = prompt + image_urls
    else:
        content = prompt

    messages = []
    messages.append({"role": "user", "content": content})
    return messages

def save_data(data, path):
    with jsonlines.open(path, mode="a") as writer:
        writer.write(data)

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            data.append(json.loads(line))
    return data

def calc_sigma(vlm, question, image, answer, predict_answer):
    image_dir_name = os.path.dirname(image)

    images = []
    idx = 1
    while True:
        image_path_next = os.path.join(image_dir_name, f"next_point_{idx}.jpg")
        idx += 1
        if not os.path.exists(image_path_next):
            break
        images.append(image_path_next)
    if len(images) > 1:
        images = images[-1:]

    system_prompt = """
[system]:
You are an AI assistant who will help me to evaluate the response given the question, the correct answer and the scene observed by the robot.
[user]:
The input includes the Question, the Answer, the Response given by the model and the Image of the environmen. You need to evaluate the alignment between the Response and the Image, as well as between the Response and the Answer, and assign a score for each.
First, assess whether the Response depends on the observed environment Image and assign one of three possible scores [0, 0.5, 1]. If the target object referenced in the Question or the Answer is present in the Image and is described accurately, assign a score of 1. If the object is present but inaccurately described, assign a score of 0.5. If the object does not exist in the Image, meaning the answer is entirely unrelated to the Image and fabricated, assign a score of 0.
Additionally, compare the model's Response with the Answer and Image, assigning a score scale from 1 to 5 based on its accuracy.
Here are some examples illustrating the degree to which response align with the correct answer, accompanied by an explanation of the score provided in parentheses.

Question: There will be 4 guests. Are there enough chairs around the dining table?
Answer: Yes, there are 6 tables.
Response: Yes.
Your mark: 5 (Correct answer. Giving a specific number is not necessary for this question.)

Question: What color is the sofa in the living room?
Answer: It is light beige.
Response: White.
Your mark: 4(The output is close to the answer but deviates.)

Question: Are the curtains in the living room closed?
Answer: No, the curtains are partially open.
Response: Yes, the curtains are closed.
Your mark: 3(The output is close to the answer but deviates because the curtain is not completely closed.)

Question: Can you tell me where the light switch for the basement is?
Answer: It is on the wall near the entrance door.
Response: The light switch on the wall near the door.
Your mark: 5(The output is completely correct.)

Question: What could I do if I get cold in the living room?
Answer: You can use the blanket on the couch next to the window.
Response: You can turn on the fireplace.
Your mark: 5(The response is inconsistent with the answer but consistent with common sense, and a fireplace can be observed in the image.)

Question: Are there any plants in the living room?
Answer: Yes, there is a plant near the sofa.
Response: No.
Your mark: 1(The output is the opposite of the answer.)

Question: What is the blue item on the bed in the nursery?
Answer: It's a baby blanket.
Response: It's a coat.
Your mark: 2(Object identification error.)

Your output should consist of exactly two fractions, separated by a comma. No further elaboration is necessary. Please provide the output that fulfills these criteria given the input.
Input:
"""

    ex_prompt = f"Question: {question}\nAnswer: {answer}\nResponse: {predict_answer}\nYour mark: "

    # result = requests_api(image_path, system_prompt + ex_prompt, max_retries=10)
    result = requests_api(images, system_prompt + ex_prompt, max_retries=10)

    # messages = [{"role": "user", "content": system_prompt + "\n" + ex_prompt}]
    # result = vlm.call_vlm(messages, image_paths=images)

    result = result.split(",")
    grd, acc = float(result[0].strip()), int(result[1].strip())

    return grd, acc

def load_jsons(files_path):
    """
    读取 JSON 文件。

    :param file_path: 文件路径
    :return: JSON 数据
    """
    all_data = []
    for file_path in files_path:
        print(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        else:
            file_ext = os.path.splitext(file_path)[-1]
            if file_ext == ".json":
                with open(file_path, 'r', encoding='utf-8') as file:
                    all_data.extend(json.load(file))
            elif file_ext == ".jsonl":
                with jsonlines.open(file_path, mode="r") as reader:
                    for item in reader:
                        all_data.append(item)
    
    return all_data

def distance(pt1, pt2):
    return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)

def evaluate(files_path, output, data_source):
    gpt = GPTEngine()
    samples = load_jsons(files_path)

    results = []
    all_path_length = 0
    print("all sample: ", len(samples))

    with open(data_source, "r") as f:
        expressbench_data = json.load(f)
    
    expressbench = {}
    for item in expressbench_data:
        idx = item['sample_id']
        expressbench[idx] = item

    for data in tqdm(samples):

        meta_data = data['meta']
        summary_data = data["summary"]
        step_data = data['step']
        
        shortest_length = expressbench[meta_data['sample_id']]['traj_length']
        question = meta_data['question']
        end_pos = expressbench[meta_data['sample_id']]["goal_position"]

        path_length = summary_data["shorest_path"]
        all_path_length += summary_data["shorest_path"]
        print(shortest_length, path_length)
        result = {}

        answer = meta_data['answer']
        predict_answer = summary_data["final_answer"]

        # step_image = [step['image']for step in step_data]
        if len(step_data) == 0:
            continue
        sigma, delta = calc_sigma(gpt, question, step_data[-1]['image'], answer, predict_answer)
        d_T = distance(step_data[-1]['pts'], end_pos)
        C = (sigma * delta) / 5
        e_path = C * shortest_length / max(path_length, shortest_length)

        result["len_steps"] = len(step_data)
        result["sigma"] = sigma
        result["delta"] = delta
        result["d_T"] = d_T
        result["C"] = C
        result["e_path"] = e_path

        save_data(result, output)

        results.append(result)

def metrics(path):
    with open("data/EXPRESS-Bench/express-bench-reacteqa.json", "r") as f:
        expressbench_data = json.load(f)

    # expressbench = {}
    # question_type = []
    # for item in expressbench_data:
    #     idx = item['sample_id']
    #     expressbench[idx] = item

    #     if item["question_type"] == "knowledge":
    #         knowledge_id.append(idx)

    results = load_jsonl(path)
    results_num = len(results)
    print(results_num)

    e_path = 0
    C = 0
    C_star = 0
    d_T = 0
    len_step = 0
    for idx, result in enumerate(results):
        len_step += result['len_steps']
        # if (idx + 1) not in knowledge_id:
        e_path += result["e_path"]
        C += result["C"]
        C_star += (result["delta"] / 5)
        d_T += result["d_T"]

    print("avg step:", len_step / results_num)
    print("========================================")
    print("E_path: ", e_path / results_num)
    print("C: ", C / results_num)
    print("C*: ", C_star / results_num)
    print("d_T: ", d_T / results_num)
    # print("E_path: ", e_path / (results_num - len(knowledge_id)))
    # print("C: ", C / (results_num - len(knowledge_id)))
    # print("C*: ", C_star / (results_num - len(knowledge_id)))
    # print("d_T: ", d_T / (results_num - len(knowledge_id)))

if __name__ == '__main__':
    files_path = [
        "results/gpt4o.zs.ov.expressbench.0921/result_4.jsonl",
        "results/gpt4o.zs.ov.expressbench.0921/result_5.jsonl",
        "results/gpt4o.zs.ov.expressbench.0921/result_6.jsonl",
        "results/gpt4o.zs.ov.expressbench.0921/result_7.jsonl",
    ]
    output = "results/gpt4o.zs.ov.expressbench.0921/result.jsonl"
    data_source = "data/EXPRESS-Bench/express-bench-reacteqa.json"
    evaluate(files_path, output, data_source)
    metrics(output)
