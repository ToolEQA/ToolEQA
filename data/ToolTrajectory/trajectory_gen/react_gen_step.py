# 通过这个脚本实现thought code observation的生成
import os
import json
from collections import defaultdict
from data.ToolTrajectory.generator_deerapi import requests_api
from src.tools.tool_box import get_tool_box, show_tool_descriptions
import re

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

def parse_blocks(response_text, filename="output.json"):
    """
    把 response 文本按 block 提取成合法 dict 列表，并保存为 json。
    """
    results = []

    # 去掉外层 ```json 和 ```
    response_text = re.sub(r"^```json", "", response_text.strip(), flags=re.MULTILINE)
    response_text = re.sub(r"```$", "", response_text.strip(), flags=re.MULTILINE).strip()

    # 切出每个对象块，用大括号对齐
    blocks = re.findall(r"\{(.*?)\}", response_text, re.DOTALL)

    for block in blocks:
        item = {}

        # thought
        m = re.search(r'"thought"\s*:\s*"([^"]+)"', block)
        if m:
            item['thought'] = m.group(1)

        # code
        m = re.search(r'"code"\s*:\s*"([^`]+)`*', block, re.DOTALL)
        if m:
            code_content = m.group(1)
            # 清理多余的 ```py 和 ```
            code_content = code_content.replace("```py", "").replace("```", "").strip()
            item['code'] = code_content

        # observation
        m = re.search(r'"observation"\s*:\s*"([^"]+)"', block, re.DOTALL)
        if m:
            item['observation'] = m.group(1)

        if item:
            results.append(item)
        
        print(results)

    # 保存成 json 文件
    # with open(filename, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=4, ensure_ascii=False)

    # print(f"✅ 已保存到 {filename}")
    # return results




def gen_react(data_path, system_prompt_path, user_prompt_path):
    system_prompt = get_prompt(system_prompt_path)
    

    tools = get_tool_box()
    tools_desc = show_tool_descriptions(tools)
    system_prompt.replace("<<tool_descriptions>>", tools_desc)
    
    user_prompt = get_prompt(user_prompt_path)
    user_prompt.replace("<<tool_descriptions>>", tools_desc)

    data = load_data(data_path)
    proposal_choice = ["A", "B", "C", "D"]

    for index, item in enumerate(data):
        # if index < 2:
        #     continue
        question = item["question"]
        choices = item["proposals"]
        answer = choices[proposal_choice.index(item["answer"][0].upper())]

        traj = item["trajectory"]

        # 提取所有步骤中的关键步骤
        steps = []
        for traj_i in traj:
            steps.append(traj_i["step"])
        
        max_suffix = defaultdict(int)

        for item in steps:
            prefix, suffix = item.split('-')
            suffix = int(suffix)
            if suffix > max_suffix[prefix]:
                max_suffix[prefix] = suffix
        # 将关键步骤放置到steps里面，以便判断与查询
        steps = []
        for i,j in max_suffix.items():
            steps.append(str(i)+'-'+str(j))
        # 找到关键物体的个数
        obj_num = int(max(list(max_suffix.keys())))
        
        for traj_i in traj:
            step_i = traj_i["step"]
            found = "False"
            all_found = "False"
            if step_i  in steps: # 如果是最后一步的话，则需要告诉gpt这是最后一步了，需要思考调用什么工具来回答问题
                images = os.path.join("/mynvme1/EQA-Traj-0611/", traj_i["image_path"])

                found = "True"
                prefix, suffix = step_i.split('-')
                if int(prefix) == obj_num: # 如果是最后一个物体，那么需要将all_found 变为true，来帮助标识
                    all_found = "True"
                print('found', found, 'all_found', all_found) 
                user_prompt_r = user_prompt.replace("<<QUERY>>", question).replace("<<TRAJECTORY>>", str(traj_i)).replace("<<FOUND>>", found).replace("<<ALL_FOUND>>", all_found)
                # print(user_prompt_r)
                print('=================================================Current Step:' + step_i +'==============================================================')
                response = requests_api(images, user_prompt_r)
                print(response["choices"][0]["message"]["content"])
                
            else: # 代表 中间的过程只是需要调用 gonextpoint工具
                images = os.path.join("/mynvme1/EQA-Traj", traj_i["image_path"])
                user_prompt_r = user_prompt.replace("<<QUERY>>", question).replace("<<TRAJECTORY>>", str(traj_i)).replace("<<FOUND>>", found).replace("<<ALL_FOUND>>", all_found)
                # response = requests_api(images, user_prompt_r)
                # print(response["choices"][0]["message"]["content"])
                # print('=================================================Current Step:' + step_i +'==============================================================')


if __name__=="__main__":
    system_prompt_path = "/home/zml/algorithm/ReactEQA/data/ToolTrajectory/prompts/trajectory/system_prompt.txt"
    user_prompt_path = "/home/zml/algorithm/ReactEQA/data/ToolTrajectory/prompts/trajectory/user_prompt_step_3.txt"
    # data_path = "/mynvme1/EQA-Traj/trajectory.json"
    data_path = "data.json"
    # gen_react(data_path, system_prompt_path, user_prompt_path)

    response_text = "[
        {
            "thought": "I have found the required target object and all required objects are fully processed",
            "code": "objectLocation2D(object='radiator', image, path=' /mynvme1/EQA-Traj-0611/vtuutd1bQeKK7KJaCHD9yw/1')",
            "observation": "The bounding box of the radiator is [x,y,z].",
        },
        {
            "thought": "Now I need to crop and save the radiator's region for further analysis.",
            "code": "objectcrop(bounding, box = [x,y,z],image_ path='/mynvme1/EQA-Traj-0611/vtuutd1bgeKK7KJaCHD9yw/1-6.')",
            "observation": "cropped radiator saved at /mynvme1/E0A-Traj-0611/vtuutd1bgekK7KJaCHD9yw/radiator-rop.png",
        }
    ]

    parse_blocks(response_text)



