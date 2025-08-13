import string
import json
import argparse



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
    traj_step = []

    for item in data["trajectory"]:
        if item["is_key"] == "true":
            react_key.append(item["react"])
            traj_step.append(item["step"])
    return react_key, traj_step



def extract_code_tool(react): 
    # 提取code中tool的列表
    tool_list = []
    for block in react:

    # block = react

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
        return False

    # # 找开始引号
    # quote_start = start_pos
    # while code[quote_start] not in ("'", '"'):
    #     quote_start += 1
    # quote_char = code[quote_start]
    # # 找结束引号
    # quote_end = quote_start + 1
    # while code[quote_end] != quote_char:
    #     quote_end += 1
    
    # extracted = code[quote_start + 1:quote_end]

    # 找括号的起始位置
    open_paren = code.find("(", start_pos)
    close_paren = code.rfind(")")
    if open_paren == -1 or close_paren == -1 or close_paren <= open_paren:
        return False  # 括号不匹配

    extracted = code[open_paren + 1:close_paren].strip()
    # 去掉包裹的引号
    if (len(extracted) >= 2) and (extracted[0] == extracted[-1]) and extracted[0] in "'\"":
        extracted = extracted[1:-1]

    final_answer_true_or_false = _normalize(extracted) == _normalize(expected)

    if final_answer_true_or_false == True:  
        return True
    else:
        print('sample_id', record["sample_id"], "expected_answer", _normalize(expected), "current_answer", _normalize(extracted))
        print('final_code', code)
        return False



def main():
    parser = argparse.ArgumentParser(description="读取 JSONL 文件并逐行处理")
    parser.add_argument("--file", type=str, required=False, help="JSONL 文件路径", default = '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/output/special.jsonl')
    parser.add_argument('--output_file', type=str, required=False, help="输出的文件路径", default = '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/output/special.json')
    args = parser.parse_args()

    fully_true = []
    order_false = []
    tool_false = []
    question_type_false = []
    final_answer_true = []
    final_answer_false = []
    data_number = 0

    wrong_data = []


    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            react_key_list,step_list = extract_react_key(data)
            object_name = []
            question_sample_id = data["sample_id"]
            question_type = data["question_type"]
            
            for item_obj in data["related_objects"]:
                object_name.append(item_obj["name"])
            
            data_number = data_number + 1
            for idx, react in enumerate(react_key_list):
                final_step = False
                if idx == len(react_key_list) - 1: # 判断是否是 最后一个元素
                    # print("最后一个:", react)
                    final_step = True
                
                # print('final_step', final_step)
                # print('react', react)
                tool_list = extract_code_tool(react)

                wrong_or_true = check_tool_order(tool_list, final_step, question_type, len(set(object_name))) 
                if wrong_or_true == "true":
                    fully_true.append(question_sample_id)
                elif wrong_or_true == "Order Wrong":
                    
                    all_keys = [key for d in wrong_data for key in d.keys()]
                    if question_sample_id in all_keys:
                        idx_key = all_keys.index(question_sample_id)
                        w_step_list_c = wrong_data[idx_key][question_sample_id]
                        w_step_list_c.append(step_list[idx])

                        wrong_data[idx_key][question_sample_id] = w_step_list_c
                    else:
                        item={}
                        w_step_list = []
                        w_step_list.append(step_list[idx])
                        item[question_sample_id] = w_step_list
                        wrong_data.append(item)
                        

                    order_false.append(question_sample_id)
                    print('sample_id', question_sample_id, 'false_step', step_list[idx], 'is_final', final_step)
                    # break
                elif wrong_or_true == "Tool Wrong":
      
                    all_keys = [key for d in wrong_data for key in d.keys()]
                    if question_sample_id in all_keys:
                        idx_key = all_keys.index(question_sample_id)
                        w_step_list_c = wrong_data[idx_key][question_sample_id]
                        w_step_list_c.append(step_list[idx])
                        wrong_data[idx_key][question_sample_id] = w_step_list_c
                    else:
                        item={}
                        w_step_list = []
                        w_step_list.append(step_list[idx])
                        item[question_sample_id] = w_step_list
                        wrong_data.append(item)
                        

                    tool_false.append(question_sample_id)
                    print('sample_id', question_sample_id, 'false_step', step_list[idx], 'is_final', final_step)
                    # break
                elif  wrong_or_true == "Question Wrong":
                    question_type_false.append(question_sample_id)
                    print('sample_id', question_sample_id)
                    break




            # final_answer_true_or_false = compare_expected_with_final_simple(data)
            # if final_answer_true_or_false:
            #     final_answer_true.append(question_sample_id)
            # else:
            #     final_answer_false.append(question_sample_id)

    print('Order Wrong List', order_false)
    print('Tool Wrong List', tool_false)
    print('question_type_false', question_type_false)

    print('final_answer_false', final_answer_false)

    # print('wrong_number', len(set(final_answer_false+tool_false+order_false)), 'all_number', data_number)
    print('wrong_number', len(wrong_data), 'all_number', data_number)

    print('wrong_data', wrong_data)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(wrong_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
