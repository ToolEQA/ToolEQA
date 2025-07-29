# 通过这个脚本实现thought code observation的生成
# color的代码和其他的代码不一样，所以不可以直接复用，因为其VQA需要两个路径
import os
import json
from collections import defaultdict
from data.ToolTrajectory.generator_deerapi import requests_api
from src.tools.tool_box import get_tool_box, show_tool_descriptions
import re
from pathlib import Path
import sys

from src.tools.vqa import VisualQATool
from src.tools.location_2d import ObjectLocation2D
from src.tools.location_3d import ObjectLocation3D
from src.tools.go_next_point import GoNextPointTool
from src.tools.segment_instance import SegmentInstanceTool
from src.tools.final_answer import FinalAnswerTool
from src.tools.crop import ObjectCrop
from pydantic import BaseModel


class Planing(BaseModel):
    plan: str

class Step(BaseModel):
    thought: str
    code: str
    observation: str

class React(BaseModel):
    steps: list[Step]


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

def load_and_split_data(data, task_id, total_tasks=3):
    # 1. 加载 JSON 数据

    data = data[:100]

    # 3. 计算每个任务的分片范围（均匀分割）
    total = len(data)
    per_task = total // total_tasks
    remainder = total % total_tasks

    start = task_id * per_task + min(task_id, remainder)
    end = start + per_task + (1 if task_id < remainder else 0)

    return data[start:end]  # 返回这一份数据

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


def parse_blocks(response_text, object_current, question_current, action = None, ObjectLocation_path = None, VisualQATool_path = None, object_information = None, expected_answer = None):
    """
    把 response 文本按 block 提取成合法 dict 列表，并保存为 json。
    """
    results = []

    results_nonprocess = []

    # 去掉外层 ```json 和 ```
    # response_text = re.sub(r"^```json", "", response_text.strip(), flags=re.MULTILINE)
    # response_text = re.sub(r"```$", "", response_text.strip(), flags=re.MULTILINE).strip()

    # 切出每个对象块，用大括号对齐
    # blocks = re.findall(r"\{(.*?)\}", response_text, re.DOTALL)
    # blocks = blocks_extract(response_text)

    for block in response_text:
        item = {}

        item_nonprocess = {}

        thought_part = block["thought"]

        code_part = block["code"]

        observation_part = block["observation"]

        # thought
        item['thought'] = thought_part
        item_nonprocess['thought'] = thought_part

        # code
        code_content = code_part

        

            
        keytool = detect_tool_or_closest(code_content)
        item_nonprocess["code"] = code_content
        if keytool is not None:
            if keytool == 'GoNextPointTool':
                if action is not None:
                    item["code"] = 'path = GoNextPointTool("' + action + '")\n' + "print(f'In this point, the current landscape is saved in {path}.')"
                else:
                    action = "move_forward"
                    item["code"] ='path = GoNextPointTool("' + action + '")\n' + "print(f'In this point, the current landscape is saved in {path}.')"
                    print('Error! Should subplace but donot provide action!')
            elif keytool == 'ObjectLocation2D':
                item["code"] = 'bbox = ' + keytool + "(object='{object_name}', image_path='{path_input}')" + "\n" + "print(f'The bounding box of " + object_current + " is {{bbox}}.')"
            elif keytool == 'ObjectLocation3D':
                item["code"] = "position, size, rot = " + keytool + "(object='{object_name}', image_path= '{path_input}')" +  "\n" + "print(f'The information of " + object_current + " is: position is {{position}},  size (Length, width, height) is {{size}}.')"
            elif keytool == 'VisualQATool':
                # color的这个和别的不一样，因为需要两个图片，所以其代码不可以复用！！！
                item["code"] = "question = '" + question_current + "' \n" + "answer = " + keytool + "(question = question, image_paths = [{path_input}])"  +  "\n" + "print(f'{{question}} {{answer}}')"
            elif keytool == 'SegmentInstanceTool':
                item["code"] = 'path = ' + code_content +  "\n" + "print(f'The semantic segmentation of " +  object_current + " is saved in {{path}}.')" 
            elif keytool == 'final_answer':
                item["code"] = code_content
                # item["code"] = keytool + 
            elif keytool == 'ObjectCrop':
                item["code"] = 'path = ' + keytool + "(bounding_box = [x1,y1,x2,y2], image_path='{path_input}')" + "\n" + "print(f'The cropped result of " +  object_current + " is saved in {{path}}.')" 
            else:
                item["code"] = code_content
        else:
            item["code"] = code_content
            print('Error! The tool is wrong!')
        
        # 为了快速适用于不同的问题类型，这里 把 对 item["code"]加上路径的方式 放到 外面的一个函数
        item["code"] = code_refine(item["code"], keytool, ObjectLocation_path, VisualQATool_path, object_information)
        
        if item["code"] is None and keytool == 'VisualQATool': # 这个表示是 返回的是None,所以直接延用之前的code_content
            item["code"] = "answer = " + code_content  +  "\n" + "print(f'{question} {answer})'"

        # observation

        item['observation'] = observation_part
        item_nonprocess['observation'] = observation_part

        results.append(item)
        results_nonprocess.append(item_nonprocess)
         


    return results, results_nonprocess
         
def code_refine(code, keytool, ObjectLocation_path = None, VisualQATool_path = None, object_information = None):
    print('code', code, "ObjectLocation_path", ObjectLocation_path)
    if keytool == 'ObjectLocation2D':
        code = code.format(object_name = object_information["name"], path_input = ObjectLocation_path)
    # elif keytool == 'GoNextPointTool':
    #     code = code.format(path = gonextpoint_path)
    elif keytool == 'ObjectCrop':
        code = code.format(path_input = ObjectLocation_path)
    elif keytool == 'VisualQATool':
        if VisualQATool_path is None:
            return None
        else:
            code = code.format(path_input = VisualQATool_path)
    
    
    return code

def parse_block_nonkey(response_text, action):
    results = []
    item = {}
    item["thought"] = response_text
    # item["code"] = 'GoNextPointTool("' + action + '")'
    item["code"] = 'path = GoNextPointTool("' + action + '")\n' + "print(f'In this point, the current landscape is saved in {path}.')"
    # item["code"] = item["code"].format(path = gonextpoint_path)
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


def extract_object_size(scene_id, object_num, data_path = "/data/zml/datasets/EmbodiedQA/HM3D"):


     # 提取内部的场景名（如 FxCkHAfgh7A）
    inner_id = scene_id.split("-")[-1]
    
    # 构建 scene 目录路径
    scene_dir = os.path.join(data_path, scene_id)

    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    # 构建 json 文件路径
    json_filename = f"{inner_id}.objects.cleaned.json"
    json_path = os.path.join(scene_dir, json_filename)
    
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # 读取 JSON 文件
    with open(json_path, "r") as f:
        objects = json.load(f)


    results = []
    size_info_pure = {}

    for obj_id in object_num:
        found = next((obj for obj in objects if obj.get("object_id") == obj_id), None)
        if found:
            # 提取类别和尺寸信息
            category = found.get("category_name", "unknown")
            dims = found.get("dimensions", [0, 0, 0])  # [length, width, height] 假设顺序

            # 构建描述字符串
            results_size = "The length, width and height of Object {} is {}, {}, and {}. ".format(
                category, dims[0], dims[2], dims[1]
            )
            results.append(results_size)
            size_info_pure[obj_id] = [dims[0], dims[2], dims[1]]
        else:
            print('Error!!! Cannot find the related object with ID:', obj_id)
            results.append("")

    # 拼接所有结果字符串
    object_size_info = "".join(results)

    return object_size_info, size_info_pure


def gen_react(data_path, system_prompt_path, planing_prompt_path, user_prompt_path, nonkey_user_prompt_path, output_path, task_id = 0):
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
    data = load_and_split_data(data, task_id) # 分割自己进程的数据

    data = pre_process(data) # 去除掉没用的thought, code, observation; 加上 "react": []
    proposal_choice = ["A", "B", "C", "D"]

    if not os.path.exists(output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f)

    for index, item in enumerate(data):
        if index > 100:
            exit()
        question = item["question"]
        choices = item["proposals"]
        answer = choices[proposal_choice.index(item["answer"][0].upper())]
        locations_infor = item["related_objects"]
        scene = item["scene"]

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
        
        objects_id = []
        for item_object in locations_infor:
            objects_id.append(int(item_object["id"]))
        
        object_pos = []
        for item_object in locations_infor:
            object_pos.append(item_object["pos"])
    

        # 获取整体问题的 plan
        object_order = "->".join(objects_name) 
        images = None
        planing_prompt_r = planing_prompt.replace("<<QUERY>>", question).replace("<<TRAJECTORY>>", object_order)
        response_plan = requests_api(images, planing_prompt_r)
        item["plan"] = response_plan
        print('response_plan', response_plan)

        # 获得关键步骤和非关键步骤的react

        react_key_results = [] # 用于保存关键的步骤的react信息
        object_information_all = []
        successful = True
        VisualQATool_path_list = []
        for traj_i, traj_i_next in zip(traj, traj[1:] + [None]):
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
                object_id_current = objects_id[int(step_i[0])]

                object_pos_current = object_pos[int(step_i[0])]
                object_pos_infor = "The position of Object {} is {}. ".format(object_name_current, str(object_pos_current))
                object_size_infor, size_info_pure = extract_object_size(scene ,[object_id_current])
                if object_size_infor == "":
                    print('Can Not Find the Objects! Cast It!')
                    successful = False
                    break

                object_information = object_pos_infor + object_size_infor
                object_information_item = {}
                object_information_item["name"] = object_name_current
                object_information_item["position"] = str(object_pos_current)
                print("size_info_pure", size_info_pure)
                object_information_item["size"] = str(size_info_pure[object_id_current])

                object_information_all.append(object_information)

                prefix, suffix = step_i.split('-')
                if int(prefix) == obj_num: # 如果是最后一个物体，那么需要将all_found 变为true，来帮助标识
                    all_found = "True"
                print('found', found, 'all_found', all_found) 
                user_prompt_r = user_prompt.replace("<<QUERY>>", question).replace("<<TRAJECTORY>>", str(traj_i)).replace("<<FOUND>>", found).replace("<<ALL_FOUND>>", all_found).replace("<<FOUND_OBJECT>>", object_name_current)
                user_prompt_r = user_prompt_r.replace("<<INFORMATION_OBJECT>>", object_information)
                # print(user_prompt_r)
                if all_found == "True":
                    user_prompt_r = user_prompt_r.replace("<<expected_answer>>", answer)
                    user_prompt_r = user_prompt_r.replace("<<previous_thought>>", str(react_key_results))
                    user_prompt_r = user_prompt_r.replace("<<provided_positions>>", str(object_information_all))

                response = requests_api(images, user_prompt_r, React)
                if all_found == "True": # 如果 all_found的话，那么不需要给action
                    action_current = None
                else:
                    if len(traj_i["action"]) == 0: # 判断一下是否为空，为空的话，则需要给一个默认值，即move_forward
                        action_current = "move_forward"
                    else:
                        action_current = traj_i["action"][0][0]
                
                if traj_i_next is None: #最后一步了，所以 一定没有 gonextpoint了。并且，else中只是 gonextpoint，所以一定要传递 图片路径。 
                    gonextpoint_path = None
                    image_path_current = "/".join(Path(traj_i["image_path"]).parts[-2:])
                    ObjectLocation_path = image_path_current

                    VisualQATool_path = "./cache/{object_name_current}-crop.png".format(object_name_current = object_name_current)
                    VisualQATool_path_list.append("'"+VisualQATool_path+"'") # 即使是最后一步了, 所以需要讲list中加上之前的path
                    VisualQATool_path_final = ", ".join(VisualQATool_path_list) # 需要构架这个，来传递
                else:
                    image_path_current = "/".join(Path(traj_i["image_path"]).parts[-2:])
                    image_path_next = "/".join(Path(traj_i_next["image_path"]).parts[-2:])
                    gonextpoint_path = image_path_next
                    ObjectLocation_path = image_path_current

                    VisualQATool_path = "./cache/{object_name_current}-crop.png".format(object_name_current = object_name_current)
                    VisualQATool_path_list.append("'"+VisualQATool_path+"'") # 因为 不是最后一步了, 所以需要讲list中加上之前的path
                    VisualQATool_path_final = None # 没到最后一步，所以不需要构建
                
                # 这个地方其实不用区分，是否是最后一步不用区分，因为如果是 
                react_results, react_results_nonprocess = parse_blocks(response, object_name_current, question, action_current, ObjectLocation_path, VisualQATool_path_final, object_information_item, answer) # 主要 要修改的是就是 路径，每一个任务所需要的路径是不一样的
                
                
                print('react_results', react_results)
                traj_i = write_in_json(traj_i, react_results)
                traj_i["is_key"] = "true"
                if all_found == "False":
                    react_key_results.append(react_results_nonprocess)
                    
                
            else: # 代表 中间的过程只是需要调用 gonextpoint工具
                print('=================================================Current Step:' + step_i +'==============================================================')
                # images = traj_i["image_path"].replace("/mynvme1/EQA-Traj-0611/", "/mynvme1/EQA-Traj-0720/")
                # user_prompt_r = user_prompt.replace("<<QUERY>>", question).replace("<<TRAJECTORY>>", str(traj_i)).replace("<<FOUND>>", found).replace("<<ALL_FOUND>>", all_found)
                images = os.path.join("/mynvme1/EQA-Traj-0720/", traj_i["image_path"])
                object_name_current = objects_name[int(step_i[0])]
                if len(traj_i["action"]) == 0: # 判断一下是否为空，为空的话，则需要给一个默认值，即move_forward
                    action_current = "move_forward"
                else:
                    action_current = traj_i["action"][0][0]
                print('object_name_current', object_name_current, 'action_current' ,action_current)
                nonkey_prompt_r = nonkey_prompt.replace("<<object_name>>", object_name_current).replace("<<action name>>", action_current) 
                response = requests_api(images, nonkey_prompt_r)
            
                response = requests_api(images, nonkey_prompt_r)
                image_path_next = "/".join(Path(traj_i_next["image_path"]).parts[-2:])
                gonextpoint_path = image_path_next
                react_results = parse_block_nonkey(response, action_current)
                traj_i = write_in_json(traj_i, react_results)
                traj_i["is_key"] = "false"

        if successful:
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
    data_path = "data/color_attributes.json"
    
    task_id = int(sys.argv[1]) 
    output_path = f"output/color_output_ans_with_plan_part_{task_id}.json"

    gen_react(data_path, system_prompt_path, planing_prompt_path, user_prompt_path, nonkey_user_prompt_path, output_path, task_id)
