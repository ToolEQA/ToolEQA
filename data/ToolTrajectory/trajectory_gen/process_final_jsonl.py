import json

# 读取 JSONL 文件
input_path = '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/location/special/output/special.jsonl'
output_path = '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/location/special/output/special_revise.jsonl'

def open_jsonl(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)

    return data

def save_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def fix_image_paths_in_code(code_str):
    # 查找 image_paths = [ 开始的部分
    prefix = 'image_paths = ['
    if prefix in code_str:
        start = code_str.index(prefix) + len(prefix)
        end = code_str.index(']', start)
        inner = code_str[start:end].strip()
        
        # 如果不是以单引号包裹，就加上
        if not inner.startswith("'") and not inner.endswith("'"):
            inner = f"'{inner}'"
            code_str = code_str[:start] + inner + code_str[end:]
    
    return code_str

def process(data_list):
    output_data = []
    for data_path in data_list:
        data = open_jsonl(data_path)
        for item in data:
            trajectory = item.get("trajectory", [])
            for traj in trajectory:
                if str(traj.get("is_key", "false")).lower() == "true":
                    reacts = traj.get("react", [])
                    for react in reacts:
                        code = react.get("code", "")
                        if "VisualQATool" in code:
                            react["code"] = fix_image_paths_in_code(code)
                            success = True
            
            # todo: item 中需要删除的字段：related_objects.region_id, removed_indices
            output_data.append(item)

    return output_data


if __name__=="__main__":
    data_list = ["data/ToolTrajectory/trajectory_gen/location/special/output/special.jsonl"]
    output_path = "react_eqa.json"
    output_data = process(data_list)
    save_json(output_data, output_path)