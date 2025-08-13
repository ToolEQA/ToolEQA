# 通过这个脚本实现thought code observation的生成
import os
import json
from collections import defaultdict
from data.ToolTrajectory.generator_deerapi import requests_api
from src.tools.tool_box import get_tool_box, show_tool_descriptions
import re
import pandas as pd
import jsonlines
from pathlib import Path

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

def save_data(data, path):
    with jsonlines.open(path, mode="a") as writer:
        writer.write(data)

def load_data(path, output_path):
    
    already_list = []

    postfix = output_path.split(".")[-1]
    if os.path.exists(output_path) and postfix == "json":
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            already_list.append(item["sample_id"])
    elif os.path.exists(output_path) and postfix == "jsonl":
        with jsonlines.open(output_path, mode="r") as reader:
            for item in reader:
                already_list.append(item["sample_id"])

    json_data = None
    postfix = path.split(".")[-1]
    if postfix == "jsonl":
        with open(path, "r") as f:
            json_data = [json.loads(line) for line in f]
    elif postfix =="json":
        with open(path, "r") as f:
            json_data = json.load(f)

    if json_data is None:
        raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")
    
    data_list = []
    for item in json_data:
        if item["sample_id"] not in already_list:
            data_list.append(item)
        else:
            print("Already exists in output file, skip:", item["sample_id"])
    
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
        'final_answer',
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
        'FinalAnswer': 'final_answer',
        'Crop': 'ObjectCrop'
    }

    for kw, tool in keyword_map.items():
        if kw in s:
            return tool

    # 都找不到
    return None



def parse_blocks(data, object_current, question_current, action = None, gonextpoint_path = None, ObjectLocation_path = None, VisualQATool_path = None, ObjectCrop_path = None, object_information = None, expected_answer = None, expected_question = None):
    """
    把 response 文本按 block 提取成合法 dict 列表，并保存为 json。
    """
    results = []


    # 去掉外层 ```json 和 ```
    # response_text = re.sub(r"^```json", "", response_text.strip(), flags=re.MULTILINE)
    # response_text = re.sub(r"```$", "", response_text.strip(), flags=re.MULTILINE).strip()

    # 切出每个对象块，用大括号对齐
    # blocks = re.findall(r"\{(.*?)\}", response_text, re.DOTALL)
    # blocks = blocks_extract(response_text)

    for block in data:
        item = {}


        thought_part = block["thought"]

        code_part = block["code"]

        observation_part = block["observation"]

        # thought
        item['thought'] = thought_part

        # code
        code_content = code_part

            
        keytool = detect_tool_or_closest(code_content)
        item["code"] = code_content
        if keytool is not None:
            if keytool == 'GoNextPointTool':
                if action is not None:
                    observation = 'In this point, the current landscape is saved in {path}.'
                    observation = observation.format(path = gonextpoint_path)
                else:
                    observation = 'In this point, the current landscape is saved in {path}.'
                    observation = observation.format(path = gonextpoint_path)
                    print('Error! Should subplace but donot provide action!')
            elif keytool == 'ObjectLocation2D':
                observation = 'The bounding box of ' + object_current + " is {bbox}."
            elif keytool == 'ObjectLocation3D':
                observation = 'The information of ' + object_current + " is: position is {position},  size (Length, width, height) is {size}."
                observation =  observation.format(position = object_information["position"], size = object_information["size"])
            elif keytool == 'VisualQATool':
                # color的这个和别的不一样，因为需要两个图片，所以其代码不可以复用！！！
                
                observation = "{question} {answer}".format(question = expected_question, answer = expected_answer)
            elif keytool == 'SegmentInstanceTool':
                item["code"] = 'path = ' + code_content +  "\n" + "print(f'The semantic segmentation of " +  object_current + " is saved in {{path}}.')" 
                observation = observation_part
            elif keytool == 'final_answer':
                item["code"] = code_content
                observation  = observation_part
                # item["code"] = keytool + 
            elif keytool == 'ObjectCrop':
                observation = "The cropped result of " + object_current + " is saved in {path}."
                observation = observation.format(path = ObjectCrop_path)
            else:
                observation = observation_part
        else:
            observation = observation_part
            print('Error! The tool is wrong!')
        
        # 为了快速适用于不同的问题类型，这里 把 对 item["code"]加上路径的方式 放到 外面的一个函数
        # item["code"] = code_refine(item["code"], keytool, ObjectLocation_path, object_information)
        
        # if item["code"] is None and keytool == 'VisualQATool': # 这个表示是 返回的是None,所以直接延用之前的code_content
        #     item["code"] = "answer = " + code_content  +  "\n" + "print(f'{question} {answer})'"

    

        item['observation'] = observation

        results.append(item)
         
    return results

def code_refine(code, keytool, ObjectLocation_path = None, object_information = None):
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
    elif keytool == 'ObjectLocation3D':
        code = code.format(object_name = object_information["name"], path_input = ObjectLocation_path)
    
    return code

# def observation_refine(keytool, input_path = None, object_information = None):
#     if keytool == 'ObjectLocation2D':
#         observation = "print(f'The bounding box of " + object_current + " is {{bbox}}.')"
#         # observation = code.format(object_name = object_information["name"], path_input = input_path)
#     elif keytool == 'GoNextPointTool':
#         code = code.format(path = gonextpoint_path)
#     elif keytool == 'ObjectCrop':
#         code = code.format(path_input = ObjectLocation_path)
#     elif keytool == 'VisualQATool':
#         if VisualQATool_path is None:
#             return None
#         else:
#             code = code.format(path_input = VisualQATool_path)
#     elif keytool == 'ObjectLocation3D':
#         code = code.format(object_name = object_information["name"], path_input = ObjectLocation_path)
    
#     return code

              
def parse_block_nonkey(item, action, gonextpoint_path):
    # item["thought"] = response_text
    # item["code"] = 'GoNextPointTool("' + action + '")'
    # item["code"] = 'path = GoNextPointTool("' + action + '")\n' + "print(f'In this point, the current landscape is saved in {path}.')"
    item[0]["observation"] = "In this point, the current landscape is saved in {path}.".format(path = gonextpoint_path)


    return item


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

def update_answer(excel_path, data_path):


    # 1. 读取 JSON
    with open(data_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # 2. 读取 Excel
    df = pd.read_csv(excel_path)

    # 确保列名统一
    df.columns = df.columns.str.strip().str.lower()
    df["scene"] = df["scene_id"].astype(str).str.strip()
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()

    # 3. 遍历 JSON 并替换 answer
    for item in json_data:
        scene = str(item.get("scene", "")).strip()
        question = str(item.get("question", "")).strip()
        
        # 先在 Excel 里按 scene 过滤
        subset = df[df["scene"] == scene]
        # 再按 question 匹配
        match = subset[subset["question"] == question]
        
        if not match.empty:
            new_answer = match.iloc[0]["answer"]
            original_answer = item["answer"]
            item["answer"] = new_answer
            print(f"替换: scene={scene}, question={question}, original answer={original_answer}, replaced answer={new_answer}")
        else:
            print(f"未找到匹配: scene={scene}, question={question}")

    # 4. 保存回 JSON
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print("更新完成，保存为 updated_data_size.json")


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


def gen_react(data_path, system_prompt_path, planing_prompt_path, user_prompt_path, nonkey_user_prompt_path, output_path, images_root):
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    
    proposal_choice = ["A", "B", "C", "D"]
    

    for index, item in enumerate(data):

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
                # print('=================================================Current Step:' + step_i +'==============================================================')   
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
                # print("size_info_pure", size_info_pure)
                object_information_item["size"] = str(size_info_pure[object_id_current])

                object_information_all.append(object_information)
                

                prefix, suffix = step_i.split('-')
                if int(prefix) == obj_num: # 如果是最后一个物体，那么需要将all_found 变为true，来帮助标识
                    all_found = "True"


                if all_found == "True": # 如果 all_found的话，那么不需要给action
                    action_current = None
                else:
                    if len(traj_i["action"]) == 0: # 判断一下是否为空，为空的话，则需要给一个默认值，即move_forward
                        action_current = "move_forward"
                    else:
                        action_current = traj_i["action"][0][0]
                

                # 需要在这个地方设置好工作所需要的路径。具体的内容在下面的注释内容中有说明。
                if traj_i_next is None: #最后一步了，所以 一定没有 gonextpoint了。并且，else中只是 gonextpoint，所以一定要传递 图片路径。 
                    gonextpoint_path = None
                    image_path_current = "/".join(Path(traj_i["image_path"]).parts[-2:])
                    ObjectLocation_path = image_path_current
                    ObjectCrop_path = image_path_current

                    VisualQATool_path = "./cache/{object_name_current}-crop.png".format(object_name_current = object_name_current)
                    VisualQATool_path_list.append("'"+VisualQATool_path+"'") # 即使是最后一步了, 所以需要讲list中加上之前的path
                    VisualQATool_path_final = ", ".join(VisualQATool_path_list) # 需要构架这个，来传递
                    ObjectCrop_path = "./cache/{object_name_current}-crop.png".format(object_name_current = object_name_current)
                else:
                    image_path_current = "/".join(Path(traj_i["image_path"]).parts[-2:])
                    image_path_next = "/".join(Path(traj_i_next["image_path"]).parts[-2:])
                    gonextpoint_path = image_path_next
                    ObjectLocation_path = image_path_current

                    VisualQATool_path = "./cache/{object_name_current}-crop.png".format(object_name_current = object_name_current)
                    VisualQATool_path_list.append("'"+VisualQATool_path+"'") # 即使是最后一步了, 所以需要讲list中加上之前的path
                    VisualQATool_path_final = ", ".join(VisualQATool_path_list) # 需要构架这个，来传递

                    ObjectCrop_path = "./cache/{object_name_current}-crop.png".format(object_name_current = object_name_current)



                # parse_blocks 的输入是 parse_blocks(response_text, object_current, question_current, action = None, gonextpoint_path = None, ObjectLocation_path = None, VisualQATool_path = None, ObjectCrop_path = None, object_information = None, expected_answer = None, expected_question = None)
                # 根据不同的任务，查看所需要的工具是啥，然后给定所需要的输入。这个代码几乎不需要改，需要在前面的内容上 提前定义好字符串。如果不需要的工作设置为None.
                # 其中 ObjectLocation_path 表示的是 objection2d, objection3d, crop的code里面所需要的路径； 这个往往是 当前traj图片的路径
                # VisualQATool_path表示的是VQA code里面所需要的路径； 这个是有两种情况的，如果前面的工具是crop的话，那么就是需要 ./crop/objection_name，如果前面没有的话，那么就直接是图片的路径。另外VQA还需要分是输入一张图片路径还是两个，这个也需要注意
                # bjectCrop_path 表示的是 crop输出的结果，所以路径是 ./crop/objection_name。
                react_results = parse_blocks(
                    traj_i["react"],
                    object_name_current,
                    question,
                    action_current,
                    gonextpoint_path=gonextpoint_path,
                    ObjectLocation_path=ObjectLocation_path,
                    VisualQATool_path=VisualQATool_path_final,
                    ObjectCrop_path=ObjectCrop_path,
                    object_information=object_information_item,
                    expected_answer=answer,
                    expected_question=question
                )
                traj_i["is_key"] = "true"
                traj_i["react"] = react_results
                
                
            else: # 代表 中间的过程只是需要调用 gonextpoint工具
                # print('=================================================Current Step:' + step_i +'==============================================================')
                
                
                object_name_current = objects_name[int(step_i[0])]
                # action_current = traj_i["action"][0][0]
                if len(traj_i["action"]) == 0: # 判断一下是否为空，为空的话，则需要给一个默认值，即move_forward
                    action_current = "move_forward"
                else:
                    action_current = traj_i["action"][0][0]
                
                image_path_next = "/".join(Path(traj_i_next["image_path"]).parts[-2:])
                gonextpoint_path = image_path_next

                react_results = parse_block_nonkey(traj_i["react"], action_current, gonextpoint_path)
                traj_i["react"] = react_results
                # traj_i = write_in_json(traj_i, react_results)
                traj_i["is_key"] = "false"

        if successful:
            save_data(item, output_path)
            # with open(output_path, 'r', encoding='utf-8') as f:
            #     data_output = json.load(f)
            
            # data_output.append(item)

            # with open(output_path, 'w', encoding='utf-8') as f:
            #     json.dump(data_output, f, ensure_ascii=False, indent=4)
        else:
            print('Fail!!!!')
            



if __name__=="__main__":
    images_root = "/mynvme1/EQA-Traj-0720/"
    system_prompt_path = "prompts/system_prompt.txt"
    planing_prompt_path = "prompts/planing_prompt.txt"
    user_prompt_path = "prompts/size_user_prompt_step_ans1.txt"
    nonkey_user_prompt_path = "prompts/nonkey_user_prompt.txt"
    # data_path = "/mynvme1/EQA-Traj/trajectory.json"
    data_path = "output/color_output_ans_with_plan_part_2.json"
    output_path = "output/color_all.jsonl"
    excel_path = "data/size_cleaned_ans.csv"

    gen_react(data_path, system_prompt_path, planing_prompt_path, user_prompt_path, nonkey_user_prompt_path, output_path, images_root)


 
    
    
