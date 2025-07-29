# 通过这个脚本实现thought code observation的生成
import os
import json
from collections import defaultdict
from data.ToolTrajectory.generator_deerapi import requests_api
from src.tools.tool_box import get_tool_box, show_tool_descriptions
import re
import pandas as pd
import ast


def normalize_choice(choice):
    """把错误编码的字符串修复成正常的 en-dash"""
    return choice.replace("â€“", "–").replace("\u00e2\u0080\u0093", "–")

def load_excel(excel_path):
    """读取并整理 excel"""
    df = pd.read_csv(excel_path)
    df.columns = df.columns.str.strip().str.lower()
    df["scene"] = df["scene_id"].astype(str).str.strip()
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df["choices"] = df["choices"].astype(str).str.strip()
    return df

def main(excel_path, data_path, output_path, question_type):
    # 读取数据
    with open(data_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    df = load_excel(excel_path)

    updated_data = []

    for item in json_data:
        qt = item.get("question_type")
        scene = str(item.get("scene", "")).strip()
        question = str(item.get("question", "")).strip()

        subset = df[df["scene"] == scene]
        match = subset[subset["question"] == question]

        # 默认保留
        keep = True

        if question_type == "size":

            if qt == "attribute-size":
                if not match.empty:
                    row = match.iloc[0]
                    new_answer = row["answer"]

                    if new_answer == "D":
                        print(f"🗑️ 删除: scene={scene}, question={question}, answer=D")
                        keep = False  # 不保留
                    else:
                        original_answer = item["answer"]
                        item["answer"] = new_answer
                        print(f"✅ 更新 attribute-size: scene={scene}, question={question}, answer={new_answer}, original_answer={original_answer}")
                else:
                    keep = False  # 不保留
                    print(f"⚠️ attribute-size 未找到匹配: scene={scene}, question={question}")

        elif question_type == "distance":

            if qt == "distance-distance":
                if not match.empty:
                    row = match.iloc[0]
                    new_answer = row["answer"]
                    choices_str = row["choices"]
                    original_answer = item["answer"]
                    item["answer"] = new_answer

                    try:
                        # choices_list = json.loads(choices_str)
                        choices_list = ast.literal_eval(choices_str)
                        choices_list = [normalize_choice(c) for c in choices_list]
                        item["proposals"] = choices_list
                        print(f"✅ 更新 distance-distance: scene={scene}, question={question}, answer={new_answer}, proposals={choices_list}, original_answer={original_answer}")
                    except Exception as e:
                        print(f"⚠️ 解析 choices 出错: scene={scene}, question={question}, choices={choices_list}, 错误: {e}")
                else:
                    keep = False  # 不保留
                    print(f"⚠️ distance-distance 未找到匹配: scene={scene}, question={question}")

        # 如果标记为保留，写回
        if keep:
            updated_data.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 更新完成，保存为 {output_path}")



def check_all_single_locations(file_path, column='locations'):
    """
    检查 Excel 中指定列是否所有行提取出来的数字只有一个。
    提取规则：匹配 'stair (数字)' 中的数字。
    """
    df = pd.read_csv(file_path)

    for val in df[column]:
        # 确保是字符串
        if not isinstance(val, str):
            return False

        # 用正则提取括号里的数字
        matches = re.findall(r'\((\d+)\)', val)

        if len(matches) != 1:
            return False

    return True




def count_json_items(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return len(data)


def check_fields_in_json(json_path, fields):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    missing_records = []  # 记录缺少字段的条目
    for idx, item in enumerate(data):
        missing = [field for field in fields if field not in item]
        if missing:
            missing_records.append((idx, missing))

    if not missing_records:
        print(f"✅ 所有记录都包含字段 {fields}")
    else:
        print(f"⚠️ 以下记录缺少字段：")
        for idx, missing in missing_records:
            print(f"  第 {idx} 条记录缺少字段: {missing}")
        print(f"共 {len(missing_records)} 条记录缺少字段")


if __name__=="__main__":
    # data_path = "trajectory.json"
    # output_path_temp = "output/trajectory_temp.json"
    # excel_path_size = "data/size_cleaned_ans.csv"
    # question_type = "size"
    # main(excel_path_size, data_path, output_path_temp, question_type)

    # output_path_temp = "output/trajectory_temp.json"
    # excel_path_distance = "data/distance_cleaned_ans_options.csv"
    # question_type = "distance"  
    # output_path = "output/trajectory_update.json"
    # main(excel_path_distance, output_path_temp, output_path, question_type)

    result = check_all_single_locations("status.csv")
    if result:
        print("✅ 所有行的 locations 都是单元素 list。")
    else:
        print("❌ 有些行不是单元素 list。")



