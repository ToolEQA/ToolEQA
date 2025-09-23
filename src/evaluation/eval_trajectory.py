import jsonlines
import json
import os
import jieba

def load_jsons(files_path):
    """
    读取 JSON 文件。

    :param file_path: 文件路径
    :return: JSON 数据
    """
    all_data = []
    for file_path in files_path:
        print(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        else:
            file_ext = os.path.splitext(file_path)[-1]
            if file_ext == ".json":
                with open(file_path, 'r', encoding='utf-8') as file:
                    all_data.extend(json.load(file))
            elif file_ext == ".jsonl":
                with jsonlines.open(file_path, mode="r") as reader:
                    for item in reader:
                        all_data.append(item)
    
    return all_data

def evaluate(files_path):
    samples = load_jsons(files_path)
    tool_patterns = [
        "ObjectLocation2D",
        "ObjectLocation3D",
        "VisualQATool",
        "GoNextPointTool",
        "ObjectCrop",
        "SegmentInstanceTool",
        "final_answer"
    ]

    total_thought_length = 0
    total_tool_length = 0
    total_react_count = 0
    for sample in samples:
        react = sample["summary"]["react"]
        for step in react:
            total_react_count += 1
            thought = step["thought"]
            code = step["code"]

            tool = {}
            for p in tool_patterns:
                tool[p] = code.count(p)
                total_tool_length += tool[p]

            total_thought_length += len(list(jieba.cut(thought)))

    print("平均每步思考长度: ", total_thought_length / total_react_count)
    print("平均每步调用工具次数: ", total_tool_length / total_react_count)

if __name__ == '__main__':
    files_path = [
        "results/qwen.zs.mc.hmeqa.0906/result_4.jsonl",
        "results/qwen.zs.mc.hmeqa.0906/result_5.jsonl",
        "results/qwen.zs.mc.hmeqa.0906/result_6.jsonl",
        "results/qwen.zs.mc.hmeqa.0906/result_7.jsonl",
    ]
    evaluate(files_path)