# 请求deerapi生成问题

import os
from openai import OpenAI
import base64
import csv
from tqdm import tqdm
import http.client
import json
import numpy as np
import cv2
import time
from pydantic import BaseModel

def transform2JSON(parsed_result):
    # print(parsed_result)  # 打印解析结果

    # 将解析的结果存储到字典中
    results = []
    for result in parsed_result.steps:
        item = {}

        item["thought"] = result.thought
        item["code"] = result.code
        item["observation"] = result.observation

        results.append(item)

    # 将字典转换为 JSON 字符串并返回，ensure_ascii=False 允许中文字符正常显示
    return results


def convert_image_to_base64(image):
    if os.path.exists(image):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image, np.ndarray):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    return None


class Planing(BaseModel):
    plan: str

class Step(BaseModel):
    thought: str
    code: str
    observation: str

class React(BaseModel):
    steps: list[Step]


def requests_api(images, prompt, text_format = None, system=None):
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
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    # 初始化客户端
    client = OpenAI(
        api_key='sk-bty7uDUznmPRiEWdi3YaUaqUpRpwiJmt2K96E0H39UbEtvMt',
        base_url='https://api.deerapi.com/v1/',  # 这里写你代理的地址
    )

    # payload = json.dumps({
    #     "model": "gpt-4o-mini",
    #     "stream": False,
    #     "messages": messages,
    #     "max_tokens": 8192,
    #     # temperature = 0.7
    # })
    

    # headers = {
    #     'Authorization': 'sk-bty7uDUznmPRiEWdi3YaUaqUpRpwiJmt2K96E0H39UbEtvMt',
    #     'Content-Type': 'application/json'
    # }

    max_retries = 3
    retry_delay = 1  # 初始延迟1秒
    data = None

    for attempt in range(max_retries):
        try:
            # conn = http.client.HTTPSConnection('api.deerapi.com')
            # conn.request("POST", "/v1/chat/completions", payload, headers)
            # res = conn.getresponse()
            # response = client.chat.completions.create(
            if text_format is not None:
                response =  client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=messages,
                    response_format=React,
                    max_tokens=8192,
                    # temperature=0.7,
                )
                data = response.choices[0].message.parsed
                data = transform2JSON(data)
            else:
                response =  client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=8192,
                    # temperature=0.7,
                )
                data = response.choices[0].message.content
            
            break

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")
            if attempt < max_retries - 1:  # 如果不是最后一次尝试
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避，增加重试间隔
                
        finally:
            client.close()

    return data

def post_process(result):
    result = result.strip("`\n")
    split = result.split("\n")
    print(split)

    question = split[0][10:]
    options = split[1][9:].strip("[]").split("; ")
    options = [option[2:].strip() for option in options]
    answer = split[2][8:].strip("[]")[0]
    label = split[3][7:].strip("[]")

    # "scene", "floor", "question", "choices", "question_formatted", "answer", "label"
    response = {
        "floor": 0,
        "question": question,
        "choices": options,
        "question_formatted": f"{question} A) {options[0]} B) {options[1]} C) {options[2]} D) {options[3]}. Answer:",
        "answer": answer,
        "label": label,
    }
    return response

def save_csv(csv_file_path, csv_columns, generated_data):
    with open(csv_file_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in generated_data:
            writer.writerow(data)

if __name__=="__main__":
    # img = cv2.imread("tmp/test.jpg")
    data = requests_api(None, "who are you?")
    print(data)

# if __name__ == "__main__":
#     scene_root = "data/ReactEQA/sample_scene/images"
#     prompt_file = "data/ReactEQA/prompt_single.txt"

#     csv_columns = ["scene", "floor", "question", "choices", "question_formatted", "answer", "label", "source_image"]

#     generated_data = []
#     index = 0
#     scene_dir = os.listdir(scene_root)
#     scene_dir.sort()
    
#     csv_file_path = 'data/ReactEQA/question.csv'

#     finished_samples = []
#     if os.path.exists(csv_file_path):
#         with open(csv_file_path, 'r') as csvfile:
#             reader = csv.DictReader(csvfile)
#             for row in reader:
#                 finished_samples.append(row['source_image'])

#     csvfile = open(csv_file_path, "a", newline='')
#     writer = csv.DictWriter(csvfile, fieldnames=csv_columns)

#     with open(prompt_file, "r") as file:
#         prompt = file.read()
    
#     for scene in tqdm(scene_dir):
#         if index > 100:
#             break

#         index += 1
#         images_file = os.listdir(os.path.join(scene_root, scene))
#         images_file.sort()
#         images_file = [os.path.join(scene_root, scene, file) for file in images_file]
        
#         result = requests_api(images_file, prompt)
#         if result is None:
#             continue
#         result_dict = post_process(result["choices"][0]["message"]["content"])
#         result_dict["source_image"] = file
#         result_dict["scene"] = file.split("_")[0]
#         generated_data.append(result_dict)

#         # try:
#         #     result = requests_api(image_file, prompt)
#         #     result_dict = post_process(result["choices"][0]["message"]["content"])
#         #     result_dict["source_image"] = file
#         #     result_dict["scene"] = file.split("_")[0]
#         #     generated_data.append(result_dict)
#         # except:
#         #     continue

#         writer.writerow(result_dict)


# from pydantic import BaseModel
# from openai import OpenAI
# from dotenv import load_dotenv
# import json
# from textwrap import dedent

# # 加载环境变量，例如 API key 等配置信息
# load_dotenv()

# # 设置 OpenAI API 的工厂名称，默认为 "openai"
# factory = "openai"

# # 初始化 OpenAI 客户端，传入 API key 和 base URL
# client = OpenAI(
#     api_key="sk-***********************************************",  # 替换为你的 DEERAPI key
#     base_url="https://api.deerapi.com/v1/"   # 这里是DEERAPI的 base url，注意这里需要 /v1/
# )

# # 定义一个产品信息类，用于解析 API 返回的数据
# class ProductInfo(BaseModel):
#     product_name: str  # 产品名称，字符串类型
#     price: float  # 价格，浮点数类型
#     description: str  # 产品描述，字符串类型

# # 定义一个提示信息，用于请求模型返回 JSON 格式的产品信息
# product_prompt = '''根据给出的产品进行分析，按json格式用中文回答,json format:product_name, price, description.'''

# # 获取产品信息的函数，传入用户的问题
# def get_product_info(question: str):
#     # 使用 OpenAI 客户端进行聊天模型的请求
#     completion = client.beta.chat.completions.parse(
#         model="gpt-4o-2024-08-06",  # 指定使用的模型
#         messages=[
#             {"role": "system", "content": dedent(product_prompt)},  # 发送系统消息，设置模型的行为
#             {"role": "user", "content": question},  # 发送用户消息，用户提出问题
#         ],
#         response_format=ProductInfo,  # 指定返回的数据格式为 ProductInfo
#     )

#     # 返回模型解析的第一个选项的消息结果
#     return completion.choices[0].message.parsed

# # 初始化一个空的产品信息字典
# product_inform = {}

# # 定义将解析的结果转换为 JSON 的函数
# def transform2JSON(parsed_result):
#     # print(parsed_result)  # 打印解析结果

#     # 将解析的结果存储到字典中
#     product_inform["product_name"] = parsed_result.product_name
#     product_inform["price"] = parsed_result.price
#     product_inform["description"] = parsed_result.description

#     # 将字典转换为 JSON 字符串并返回，ensure_ascii=False 允许中文字符正常显示
#     return json.dumps(product_inform, ensure_ascii=False, indent=4)

# # 定义用户输入的问题，即一个产品信息的描述
# question = "75寸小米电视机"

# # 调用函数获取产品信息
# result = get_product_info(question)


# # 将解析结果转换为 JSON 格式并打印
# json_result = transform2JSON(result)
# print(json_result)