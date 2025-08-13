import string
import json
import argparse
import jsonlines


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

def extract_react_key(data):
    react_key = []

    for item in data["trajectory"]:
        if item["is_key"] == "true":
            react_key.append(item["react"])
    return react_key



def extract_code_tool(react): 
    # 提取code中tool的列表
    tool_list = []
    for block in react:
        code_part = block["code"]
        code_content = code_part    
        # print('code_content', code_content)        
        keytool = detect_tool_or_closest(code_content)
        tool_list.append(keytool)
    
    return tool_list

def all_tool_order():
    tool_order = []
    # 定义各个问题的工具顺序
    item = {}
    item["question_type"] = "attribute-color"
    item["tool_list_nonfinal"] = ["ObjectLocation2D", "ObjectCrop", "GoNextPointTool"]
    item["tool_list_final"] = ["ObjectLocation2D", "ObjectCrop", "VisualQATool", "final_answer"]
    tool_order.append(item)

    item = {}
    item["question_type"] = "attribute-size"
    item["tool_list_nonfinal"] = ["ObjectLocation3D", "GoNextPointTool"]
    item["tool_list_final"] = ["ObjectLocation3D", "final_answer"]
    tool_order.append(item)

    item = {}
    item["question_type"] = "attribute-special"
    item["tool_list_nonfinal"] = []
    item["tool_list_final"] = ["ObjectLocation2D", "ObjectCrop", "VisualQATool", "final_answer"]
    tool_order.append(item)
    
    item = {}
    item["question_type"] = "counting-counting"
    item["tool_list_nonfinal"] = ["ObjectLocation3D", "GoNextPointTool"]
    item["tool_list_final"] = ["ObjectLocation3D", "final_answer"]
    tool_order.append(item)

    item = {}
    item["question_type"] = "distance-distance"
    item["tool_list_nonfinal"] = ["ObjectLocation3D", "GoNextPointTool"]
    item["tool_list_final"] = ["ObjectLocation3D", "final_answer"]
    tool_order.append(item)

    item = {}
    item["question_type"] = "location-location-one"
    item["tool_list_nonfinal"] = ["VisualQATool","GoNextPointTool"]
    item["tool_list_final"] = ["VisualQATool", "final_answer"]
    tool_order.append(item)

    item = {}
    item["question_type"] = "location-location-two"
    item["tool_list_nonfinal"] = ["GoNextPointTool"]
    item["tool_list_final"] = ["VisualQATool", "final_answer"]
    tool_order.append(item)

    item = {}
    item["question_type"] = "location-special"
    item["tool_list_nonfinal"] = []
    item["tool_list_final"] = ["VisualQATool", "final_answer"]
    tool_order.append(item)

    item = {}
    item["question_type"] = "relationship-relationship"
    item["tool_list_nonfinal"] = []
    item["tool_list_final"] = ["VisualQATool", "final_answer"]
    tool_order.append(item)

    item = {}
    item["question_type"] = "status-status"
    item["tool_list_nonfinal"] = []
    item["tool_list_final"] = ["ObjectLocation2D", "ObjectCrop", "VisualQATool", "final_answer"]
    tool_order.append(item)
    
    return tool_order

def check_tool_order(tool_list, final_step, question_type, objects_number = 1): # 检查tool的顺序是否正确
    
    

    allreal_tool_list = [
        'VisualQATool',
        'ObjectLocation2D',
        'ObjectLocation2D',
        'GoNextPointTool',
        'SegmentInstanceTool',
        'final_answer',
        'ObjectCrop'
    ]
    
    if final_step == True:
        isfinal = "tool_list_final"
    else:
        isfinal = "tool_list_nonfinal"
    # print('isfinal', isfinal)
    
    tool_order = all_tool_order()

    if question_type == "attribute-color":
        if None in tool_list:
            return "Name Wrong"

        item = next(item for item in tool_order if item["question_type"] == "attribute-color")
        tool_order_current = item[isfinal]

        if tool_list == tool_order_current:
            return "True"
        else:
            print('Wrong!', 'tool_list', tool_list, 'tool_order_current',tool_order_current)
            if set(tool_list) == set(tool_order_current):
                return "Order Wrong"
            else:
                return "Tool Wrong"
    
    elif question_type == "attribute-size":

        item = next(item for item in tool_order if item["question_type"] == "attribute-size")
        tool_order_current = item[isfinal]

        if tool_list == tool_order_current:
            return "True"
        else:
            tool_list = [x for x in tool_list if x is not None]
            if tool_list == tool_order_current:
                return "True"
            else:
                print('Wrong!', 'tool_list', tool_list, 'tool_order_current',tool_order_current)
                if set(tool_list) == set(tool_order_current):
                    return "Order Wrong"
                else:
                    return "Tool Wrong"

    elif question_type == "attribute-special":
        if None in tool_list:
            return "Tool Wrong"
            
        item = next(item for item in tool_order if item["question_type"] == "attribute-special")
        tool_order_current = item[isfinal]

        if tool_list == tool_order_current:
            return "True"
        else:
            print('Wrong!', 'tool_list', tool_list, 'tool_order_current',tool_order_current)
            if set(tool_list) == set(tool_order_current):
                return "Order Wrong"
            else:
                return "Tool Wrong"

    elif question_type == "counting-counting":
        if None in tool_list:
            return "Tool Wrong"
            
        item = next(item for item in tool_order if item["question_type"] == "counting-counting")
        tool_order_current = item[isfinal]

        if tool_list == tool_order_current:
            return "True"
        else:
            print('Wrong!', 'tool_list', tool_list, 'tool_order_current',tool_order_current)
            if set(tool_list) == set(tool_order_current):
                return "Order Wrong"
            else:
                return "Tool Wrong"
   
    elif question_type == "distance-distance":
        
        item = next(item for item in tool_order if item["question_type"] == "distance-distance")
        tool_order_current = item[isfinal]
        tool_list = [x for x in tool_list if x is not None]
        if tool_list == tool_order_current:
            return "True"
        else:
            print('Wrong!', 'tool_list', tool_list, 'tool_order_current',tool_order_current)
            if set(tool_list) == set(tool_order_current):
                return "Order Wrong"
            else:
                return "Tool Wrong"
    
    elif question_type == "location-location":
        if None in tool_list:
            return "Tool Wrong"
            
        if objects_number == 1:
            item = next(item for item in tool_order if item["question_type"] == "location-location-one")
            tool_order_current = item[isfinal]
        else:
            item = next(item for item in tool_order if item["question_type"] == "location-location-two")
            tool_order_current = item[isfinal]
        
        if tool_list == tool_order_current:
            return "True"
        else:
            print('Wrong!', 'tool_list', tool_list, 'tool_order_current',tool_order_current)
            if set(tool_list) == set(tool_order_current):
                return "Order Wrong"
            else:
                return "Tool Wrong"

    elif question_type == "location-special":
        if None in tool_list:
            return "Tool Wrong"
            
        item = next(item for item in tool_order if item["question_type"] == "location-special")
        tool_order_current = item[isfinal]

        if tool_list == tool_order_current:
            return "True"
        else:
            print('Wrong!', 'tool_list', tool_list, 'tool_order_current',tool_order_current)
            if set(tool_list) == set(tool_order_current):
                return "Order Wrong"
            else:
                return "Tool Wrong"

    elif question_type == "relationship-relationship":
        if None in tool_list:
            return "Tool Wrong"
            
        item = next(item for item in tool_order if item["question_type"] == "relationship-relationship")
        tool_order_current = item[isfinal]

        if tool_list == tool_order_current:
            return "True"
        else:
            print('Wrong!', 'tool_list', tool_list, 'tool_order_current',tool_order_current)
            if set(tool_list) == set(tool_order_current):
                return "Order Wrong"
            else:
                return "Tool Wrong"
        
    elif question_type == "status-status":
        if None in tool_list:
            return "Tool Wrong"
            
        item = next(item for item in tool_order if item["question_type"] == "status-status")
        tool_order_current = item[isfinal]

        if tool_list == tool_order_current:
            return "True"
        else:
            print('Wrong!', 'tool_list', tool_list, 'tool_order_current',tool_order_current)
            if set(tool_list) == set(tool_order_current):
                return "Order Wrong"
            else:
                return "Tool Wrong"
    else:
        print('Wrong!', question_type)
        return "Question Wrong"

def _normalize(text: str) -> str:
    # 去掉首尾空格、引号、小写、去掉首尾标点
    t = text.strip()
    if (len(t) >= 2) and (t[0] == t[-1]) and t[0] in "'\"":
        t = t[1:-1]
    t = t.lower().strip()
    t = t.strip(string.punctuation + " ")
    return " ".join(t.split())

def compare_expected_with_final_simple(record: dict):
    proposal_choice = ["A", "B", "C", "D"]
    choices = record["proposals"]
    expected = choices[proposal_choice.index(record["answer"][0].upper())]
    code = record["trajectory"][-1]["react"][-1]["code"]

    start_pos = code.lower().find("final_answer(")
    if start_pos == -1:
        # 如果找不到 final_answer 就直接返回 False 或者 None
        print('failure code', code)
        return None


    # 找括号的起始位置
    open_paren = code.find("(", start_pos)
    close_paren = code.rfind(")")
    if open_paren == -1 or close_paren == -1 or close_paren <= open_paren:
        print('failure code', code)
        return None  # 括号不匹配

    extracted = code[open_paren + 1:close_paren].strip()
    # 去掉包裹的引号
    if (len(extracted) >= 2) and (extracted[0] == extracted[-1]) and extracted[0] in "'\"":
        extracted = extracted[1:-1]
    print('revise observation', extracted, 'original observation', record["trajectory"][-1]["react"][-1]["observation"])
    record["trajectory"][-1]["react"][-1]["observation"] = extracted

    return record



def main():
    parser = argparse.ArgumentParser(description="读取 JSONL 文件并逐行处理")
    parser.add_argument("--input_file", type=str, required=False, help="JSONL 文件路径", default = '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/output/special.jsonl')
    parser.add_argument("--output_file", type=str, required=False, help="JSONL 文件路径", default = '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/output/special_revise.jsonl')
    args = parser.parse_args()

    new_data = []
    failure_sample = []
    data_number = 0


    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            record_new = compare_expected_with_final_simple(data)
            
            if record_new is None:
                failure_sample.append(data['sample_id'])
            else:
                new_data.append(record_new)

    # with jsonlines.open(args.output_file, mode="w") as writer:
    #     writer.write_all(new_data)  # ✅ 一次写入多个 dict
    
    with jsonlines.open(args.output_file, mode="w") as writer:
        for item in new_data:
            writer.write(item)  # ✅ 一次写一个 dict


if __name__ == "__main__":
    main()
