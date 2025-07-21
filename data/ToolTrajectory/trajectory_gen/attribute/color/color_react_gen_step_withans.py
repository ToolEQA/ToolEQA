# 通过这个脚本实现thought code observation的生成
import os
import json
from collections import defaultdict
from data.ToolTrajectory.generator_deerapi import requests_api
from src.tools.tool_box import get_tool_box, show_tool_descriptions
import re

from src.tools.vqa import VisualQATool
from src.tools.location_2d import ObjectLocation2D
from src.tools.location_3d import ObjectLocation3D
from src.tools.go_next_point import GoNextPointTool
from src.tools.segment_instance import SegmentInstanceTool
from src.tools.final_answer import FinalAnswerTool
from src.tools.crop import ObjectCrop

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

def detect_tool_or_closest(s):

      # MODEL_TOOLBOX = [
    #             VisualQATool(),
    #             ObjectLocation2D(),
    #             ObjectLocation3D(),
    #             GoNextPointTool(),
    #             SegmentInstanceTool(),
    #             FinalAnswerTool(),
    #             ObjectCrop()
    #         ]

    tools = [
        'VisualQATool',
        'ObjectLocation2D',
        'ObjectLocation2D',
        'GoNextPointTool',
        'SegmentInstanceTool',
        'FinalAnswerTool',
        'ObjectCrop'
    ]

    # 严格匹配
    for tool in tools:
        if tool in s:
            return tool

    # 模糊关键词到工具名的映射
    keyword_map = {
        'Location2D': 'ObjectLocation2D',
        'Location3D': 'ObjectLocation3D',
        'VisualQA': 'VisualQATool',
        'GoNextPoint': 'GoNextPointTool',
        'SegmentInstance': 'SegmentInstanceTool',
        'FinalAnswer': 'FinalAnswerTool',
        'Crop': 'ObjectCrop'
    }

    for kw, tool in keyword_map.items():
        if kw in s:
            return tool

    # 都找不到
    return None


def parse_blocks(response_text, object_current, question_current, action = None):
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
        m = re.search(
            r'"code"\s*:\s*`{3}py\s*(.*?)`{3}',
            block,
            re.DOTALL
        )

        if m:
            code_content = m.group(1)
            # 清理多余的 ```py 和 ```
            code_content = code_content.replace("```py", "").replace("```", "").strip()
            keytool = detect_tool_or_closest(code_content)
            if keytool is not None:
                if keytool == 'GoNextPointTool':
                    if action is not None:
                        item["code"] = 'GoNextPointTool("' + action + '")\n' + "print(f'In this point, the current landscape is saved in {path}.')"
                    else:
                        action = "move_forward"
                        item["code"] ='GoNextPointTool("' + action + '")\n' + "print(f'In this point, the current landscape is saved in {path}.')"
                        print('Error! Should subplace but donot provide action!')
                elif keytool == 'ObjectLocation2D':
                    item["code"] = code_content + "\n" + "print(f'The bounding box of " + object_current + " is [x1,y1,x2,y2].')"
                elif keytool == 'ObjectLocation3D':
                    item["code"] = code_content +  "\n" + "print(f'The information of " + object_current + " is: position is {position},  size (Length, width, height) is {size}, and the rotation is {size}.')"
                elif keytool == 'VisualQATool':
                    item["code"] = code_content  +  "\n" + "print(f'The question is " + question_current + ". The answer is {answer}.')"
                elif keytool == 'SegmentInstanceTool':
                    item["code"] = code_content +  "\n" + "print(f'The semantic segmentation of " +  object_current + " is saved in {path}." 
                elif keytool == 'FinalAnswerTool':
                    item["code"] = code_content
                elif keytool == 'ObjectCrop':
                    item["code"] = code_content + "\n" + "print(f'The cropped result of " +  object_current + " is saved in {path}." 
                else:
                    item["code"] = code_content
            else:
                item["code"] = code_content
                print('Error! The tool is wrong!')
         

        # observation
        m = re.search(r'"observation"\s*:\s*"([^"]+)"', block, re.DOTALL)
        if m:
            item['observation'] = m.group(1)

        if item:
            results.append(item)
        
    return results
        

def parse_block_nonkey(response_text, action):
    results = []
    item = {}
    item["thought"] = response_text
    # item["code"] = 'GoNextPointTool("' + action + '")'
    item["code"] = 'GoNextPointTool("' + action + '")\n' + "print(f'In this point, the current landscape is saved in {path}.')"
    item["observation"] = "Navigating to the next point in the 3D environment."
    results.append(item)

    return results


def write_in_json(traj, react_results):
    # 将 react的结果写入到 json数据中
    for items in react_results:
        traj["react"].append(items)
    
    return traj

def extract_all_names(text):
    """
    输入: 一个字符串
    输出: 一个列表，包含所有名字
    """
    return re.findall(r'(\w+)\s*\(\d+\):', text)

def gen_react(data_path, system_prompt_path, planing_prompt_path, user_prompt_path, nonkey_user_prompt_path, output_path):
    system_prompt = get_prompt(system_prompt_path)
    

    tool_box_selected = [ ObjectLocation2D(debug=True), GoNextPointTool(debug=True),ObjectCrop(debug=True), VisualQATool(debug=True), FinalAnswerTool(debug=True)]
    tools = get_tool_box(debug=True, tool_box_selected = tool_box_selected)
    tools_desc = show_tool_descriptions(tools)
    print('tools_desc', tools_desc)

    system_prompt.replace("<<tool_descriptions>>", tools_desc)
    
    user_prompt = get_prompt(user_prompt_path)
    user_prompt.replace("<<tool_descriptions>>", tools_desc)

    nonkey_prompt = get_prompt(nonkey_user_prompt_path)

    planing_prompt = get_prompt(planing_prompt_path)

    data = load_data(data_path)
    data = pre_process(data) # 去除掉没用的thought, code, observation; 加上 "react": []
    proposal_choice = ["A", "B", "C", "D"]

    if not os.path.exists(output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f)

    for index, item in enumerate(data):
        if index < 6:
            continue
        question = item["question"]
        choices = item["proposals"]
        answer = choices[proposal_choice.index(item["answer"][0].upper())]
        locations_infor = item["related_objects"]

        traj = item["trajectory"]

        # 提取所有步骤中的关键步骤
        steps = []
        for traj_i in traj:
            steps.append(traj_i["step"])
        
        max_suffix = defaultdict(int)

        for item_steps in steps:
            prefix, suffix = item_steps.split('-')
            suffix = int(suffix)
            if suffix > max_suffix[prefix]:
                max_suffix[prefix] = suffix
        # 将关键步骤放置到steps里面，以便判断与查询
        steps = []
        for i,j in max_suffix.items():
            steps.append(str(i)+'-'+str(j))
        # 找到关键物体的个数
        obj_num = int(max(list(max_suffix.keys())))
        
        objects_name = []
        for item_object in locations_infor:
            objects_name.append(item_object["name"])
    

        # 获取整体问题的 plan
        object_order = "->".join(objects_name) 
        images = None
        planing_prompt_r = planing_prompt.replace("<<QUERY>>", question).replace("<<TRAJECTORY>>", object_order)
        response_plan = requests_api(images, planing_prompt_r)
        item["plan"] = response_plan["choices"][0]["message"]["content"]

        # 获得关键步骤和非关键步骤的react

        react_key_results = [] # 用于保存关键的步骤的react信息
        for traj_i in traj:
            step_i = traj_i["step"]
            found = "False"
            all_found = "False"
            if step_i  in steps: # 如果是最后一步的话，则需要告诉gpt这是最后一步了，需要思考调用什么工具来回答问题
                # images = os.path.join("/mynvme1/EQA-Traj-0611/", traj_i["image_path"])
                print('=================================================Current Step:' + step_i +'==============================================================')   
                print("action", traj_i)
                
                images = None
                found = "True"
                object_name_current = objects_name[int(step_i[0])]
                prefix, suffix = step_i.split('-')
                if int(prefix) == obj_num: # 如果是最后一个物体，那么需要将all_found 变为true，来帮助标识
                    all_found = "True"
                print('found', found, 'all_found', all_found) 
                user_prompt_r = user_prompt.replace("<<QUERY>>", question).replace("<<TRAJECTORY>>", str(traj_i)).replace("<<FOUND>>", found).replace("<<ALL_FOUND>>", all_found).replace("<<FOUND_OBJECT>>", object_name_current)
                # print(user_prompt_r)
                if all_found == "True":
                    user_prompt_r = user_prompt_r.replace("<<expected_answer>>", answer)
                    user_prompt_r = user_prompt_r.replace("<<previous_thought>>", str(react_key_results))
                response = requests_api(images, user_prompt_r)
                       
                print(response["choices"][0]["message"]["content"])
                if all_found == "True": # 如果 all_found的话，那么不需要给action
                    action_current = None
                else:
                    if len(traj_i["action"]) == 0: # 判断一下是否为空，为空的话，则需要给一个默认值，即move_forward
                        action_current = "move_forward"
                    else:
                        action_current = traj_i["action"][0][0]

                react_results = parse_blocks(response["choices"][0]["message"]["content"],  object_name_current, question, action_current)
                traj_i = write_in_json(traj_i, react_results)
                traj_i["is_key"] = "true"
                if all_found == "False":
                    react_key_results.append(react_results)
                    
                
            else: # 代表 中间的过程只是需要调用 gonextpoint工具
                print('=================================================Current Step:' + step_i +'==============================================================')
                images = os.path.join("/mynvme1/EQA-Traj-0611/", traj_i["image_path"])
                # user_prompt_r = user_prompt.replace("<<QUERY>>", question).replace("<<TRAJECTORY>>", str(traj_i)).replace("<<FOUND>>", found).replace("<<ALL_FOUND>>", all_found)
                
                object_name_current = objects_name[int(step_i[0])]
                if len(traj_i["action"]) == 0: # 判断一下是否为空，为空的话，则需要给一个默认值，即move_forward
                    action_current = "move_forward"
                else:
                    action_current = traj_i["action"][0][0]
                print('object_name_current', object_name_current, 'action_current' ,action_current)
                nonkey_prompt_r = nonkey_prompt.replace("<<object_name>>", object_name_current).replace("<<action name>>", action_current) 
                response = requests_api(images, nonkey_prompt_r)
                
                print(response["choices"][0]["message"]["content"])
                react_results = parse_block_nonkey(response["choices"][0]["message"]["content"], action_current)
                traj_i = write_in_json(traj_i, react_results)
                traj_i["is_key"] = "false"

        with open(output_path, 'r', encoding='utf-8') as f:
            data_output = json.load(f)
        
        data_output.append(item)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_output, f, ensure_ascii=False, indent=4)

            

def pre_process(data):
    for index, item in enumerate(data):

        traj = item["trajectory"]

        for traj_i in traj:
            traj_i["react"] = []
            for k in {"thought", "code", "observation"}:
                traj_i.pop(k, None)
            # for k in {"action", "rotation"}:
            #     traj_i.pop(k, None)
        if traj[0]["action"] == "":
            # 保存 traj[0] 原来的 action
            first_action = traj[0]['action']
            # 从 0 到 len(A)-2，每个元素的 action 设为下一个的
            for i in range(len(traj) - 1):
                traj[i]['action'] = traj[i+1]['action']
            # 最后一个的 action 设为原来第一个的
            traj[-1]['action'] = first_action

    return data


if __name__=="__main__":
    system_prompt_path = "prompts/system_prompt.txt"
    planing_prompt_path = "prompts/planing_prompt.txt"
    user_prompt_path = "prompts/color_user_prompt_step_ans.txt"
    nonkey_user_prompt_path = "prompts/nonkey_user_prompt.txt"
    # data_path = "/mynvme1/EQA-Traj/trajectory.json"
    data_path = "data/data_color.json"
    output_path = "output/color_output_ans_with_plan_nonkey-t.json"
    gen_react(data_path, system_prompt_path, planing_prompt_path, user_prompt_path, nonkey_user_prompt_path, output_path)

    # # 读取第一个文件
    # with open("output_ans_nonkey.json", "r", encoding="utf-8") as f:
    #     ans_list = json.load(f)

    # # 读取第二个文件
    # with open("output_ans_plan.json", "r", encoding="utf-8") as f:
    #     plan_list = json.load(f)

    # # 检查长度
    # if len(ans_list) != len(plan_list):
    #     raise ValueError("两个文件的列表长度不一致！")

    # # 把plan加进去
    # for ans_item, plan_item in zip(ans_list, plan_list):
    #     ans_item["plan"] = plan_item.get("plan")

    # # 保存到新文件
    # with open("output_ans_with_plan.json", "w", encoding="utf-8") as f:
    #     json.dump(ans_list, f, ensure_ascii=False, indent=2)

    # print("合并完成，结果保存在 output_ans_with_plan.json")

