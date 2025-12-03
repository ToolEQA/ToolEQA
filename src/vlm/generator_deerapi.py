# 请求deerapi生成问题
import os
import base64
import csv
import http.client
import json
import numpy as np
import cv2
import time

from openai import OpenAI
from tqdm import tqdm
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


def requests_api(images, prompt, text_format = None, system=None, max_retries=5):
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
        # api_key='API_KEY',
        api_key="sk-YqbuQUTS82eXW6zleSru7InNsV88G6APi68pa4IiGuUUHNox",
        base_url='https://api.deerapi.com/v1/',  # 这里写你代理的地址
    )

    retry_delay = 1  # 初始延迟1秒
    data = None

    for attempt in range(max_retries):
        try:
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
                response =  client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=8192,
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

def requests_api_deprecated(images, prompt, system=None):
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

    payload = json.dumps({
        "model": "gpt-4o-mini",
        "stream": False,
        "messages": messages,
        "max_tokens": 8192,
    })
    
    headers = {
        'Authorization': 'your api key',
        'Content-Type': 'application/json'
    }

    max_retries = 3
    retry_delay = 1  # 初始延迟1秒
    data = None

    for attempt in range(max_retries):
        try:
            conn = http.client.HTTPSConnection('api.deerapi.com')
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            if res.status == 200:
                data = json.loads(res.read().decode("utf-8"))
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")
            if attempt < max_retries - 1:  # 如果不是最后一次尝试
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避，增加重试间隔
        finally:
            conn.close()

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
    # data = requests_api(None, "who are you?", React)
    data = requests_api(None, "who are you?")
    print(data)
