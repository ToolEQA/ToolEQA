# 通过这个脚本实现thought code observation的生成
import os
import json
from data.ToolTrajectory.generator_deerapi import requests_api
from src.tools.tool_box import get_tool_box, show_tool_descriptions

def load_data(path):
    data_list = []
    postfix = path.split(".")[-1]
    if postfix == "jsonl":
        with open(path, "r") as f:
            data_list = [json.loads(line) for line in f]
    elif postfix =="json":
        with open(path, "r") as f:
            data_list = json.load(f)
    return data_list

def get_prompt(path):
    with open(path, "r", encoding="utf-8") as file:
        prompt = file.read()
    return prompt

def gen_react(data_path, system_prompt_path, user_prompt_path):
    system_prompt = get_prompt(system_prompt_path)
    user_prompt = get_prompt(user_prompt_path)

    tools = get_tool_box()
    tools_desc = show_tool_descriptions(tools)
    system_prompt.replace("<<tool_descriptions>>", tools_desc)

    data = load_data(data_path)
    proposal_choice = ["A", "B", "C", "D"]

    for index, item in enumerate(data):
        # if index < 2:
        #     continue
        question = item["question"]
        choices = item["proposals"]
        answer = choices[proposal_choice.index(item["answer"][0].upper())]

        traj = item["trajectory"]
        # print(traj)

        images = [os.path.join("/mynvme1/EQA-Traj", item["image_path"]) for item in traj]
        print(len(images), images)

        user_prompt = user_prompt.replace("<<QUERY>>", question).replace("<<TRAJECTORY>>", str(traj))#.replace("<<ANSWER>>", answer)
        print("============================")
        print(user_prompt)
        print(choices, answer)
        print("============================")
        response = requests_api(images, user_prompt, system_prompt)
        print(response["choices"][0]["message"]["content"])
        exit()

if __name__=="__main__":
    system_prompt_path = "/home/zml/algorithm/ReactEQA/data/ToolTrajectory/prompts/trajectory/system_prompt.txt"
    user_prompt_path = "/home/zml/algorithm/ReactEQA/data/ToolTrajectory/prompts/trajectory/user_prompt_3.txt"
    # data_path = "/mynvme1/EQA-Traj/trajectory.json"
    data_path = "data.json"
    gen_react(data_path, system_prompt_path, user_prompt_path)
