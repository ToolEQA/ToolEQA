# location问题，根据问题生成选项和答案

import pandas as pd
import json
import re

import random

# 读取CSV文件
def read_csv(file_path):
    return pd.read_csv(file_path)

# 读取JSON文件
def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 提取问题中的物体，可以处理多个物体的情况
def extract_objects_from_question(question):
    # 去掉不必要的空格并将句子转成小写
    question = question.strip().lower()
    
    # 存储提取出的物体
    objects = []

    # 如果问题包含 "and"，认为是多个物体，逐个提取
    if 'and' in question:
        # 拆分成两个部分，通过"and"分隔
        parts = question.split('and')
        # print(parts[0])
        # print(parts[1])
        match = re.search(r'in which rooms? are the ([a-zA-Z\s]+)', parts[0])
        # match = re.search(r'\bin which rooms? are the (\w+)', parts[0])
        if match:
            objects.append(match.group(1))

        match = re.search(r'the ([a-zA-Z\s]+?)(?=\s+located)', parts[1])
        if match:
            objects.append(match.group(1))

    else:
        # 如果没有"and"，直接提取单个物体
        # match = re.search(r'\bin which rooms? is the (\w+)', question)
        match = re.search(r'in which rooms? is the ([a-zA-Z\s]+?)(?=\s+located)', question)
    
        if match:
            objects.append(match.group(1))

    # 过滤掉无关词汇（如 in, which, rooms, is, are 等）
    objects = [obj.strip() for obj in objects if obj not in {'in', 'which', 'rooms', 'is', 'are'}]

    return objects


# 根据物体名称从JSON中查找对应的记录
def find_location_for_object(json_data, object_name):
    locations = []
    for record in json_data:
        if record['category_name'].lower() == object_name.lower():
            locations.append(record)
    return locations

# 主函数
def find_object_location_all(csv_path, json_path):
    # 读取CSV和JSON数据
    questions = read_csv(csv_path)['questions']  # 假设CSV文件有一列名为'questions'
    json_data = read_json(json_path)

    # 对每个问题进行处理
    for question in questions:
        # 提取物体名，支持单个物体和多个物体
        objects = extract_objects_from_question(question)
        print(f"Question: {question}")
        print(objects)
        
        if objects:
            print(f"Question: {question}")
            for object_name in objects:
                locations = find_location_for_object(json_data, object_name)
                if locations:
                    print(f"\nObject: {object_name.capitalize()}")
                    for location in locations:
                        print(f"Category Name: {location['category_name']}")
                        print(f"Region ID: {location['region_id']}")
                        print(f"Dimensions: {location['dimensions']}")
                        print(f"Center: {location['center']}")
                        print("-" * 40)
                else:
                    print(f"No location found for object: {object_name.capitalize()}")
        else:
            print(f"Could not extract objects from the question: {question}")

def find_object_details(object_name, json_data):
    # 存储找到的物体信息
    object_details = []
    
    for item in json_data:
        # 忽略大小写进行比较
        if item['category_name'].lower() == object_name.lower():
            object_details.append({
                'category_name': item['category_name'],
                'region_id': item['region_id'],
                'dimensions': item['dimensions']
            })
    return object_details  # 如果没有找到物体，返回空列表

def find_object_region_ids(object_name, json_data):
    # 存储找到的物体的 region_id
    region_ids = []
    
    for item in json_data:
        # 忽略大小写进行比较
        if item['category_name'].lower() == object_name.lower():
            region_ids.append(item['region_id'])
    
    return region_ids  # 如果没有找到物体，返回空列表

def get_all_region_ids(json_data):
    region_ids = []
    for item in json_data:
        region_ids.append(item['region_id'])
    return set(region_ids)

# 随机生成错误选项

import random

import random

def generate_wrong_options(correct_region_ids, all_region_ids, num_wrong=3):
    """
    生成多个错误答案，每个错误答案是一个包含多个区域的列表
    :param correct_region_ids: 正确答案的区域列表
    :param all_region_ids: 所有区域的列表
    :param num_wrong: 需要生成错误答案的数量
    :param min_regions: 每个错误答案包含的最少区域数
    :param max_regions: 每个错误答案包含的最多区域数
    :return: 错误答案列表
    """
    wrong_choices = set()  # 使用 set 来确保唯一性，set 中的元素不会重复
    
    while len(wrong_choices) < num_wrong:
        # 随机决定每个错误答案中包含的区域数量
        num_regions = random.randint(0, len(all_region_ids)-4)  # 随机选择每个错误答案的区域数
        
        # 随机决定每个错误答案中与正确答案重叠的区域数量
        overlap_count = random.randint(0, min(len(correct_region_ids), num_regions))  # 控制重叠数量
        overlap_regions = random.sample(correct_region_ids, overlap_count)  # 随机选择重叠区域
        
        # 从剩余的区域中选择其余的错误选项
        remaining_regions = list(set(all_region_ids) - set(correct_region_ids))
        remaining_count = num_regions - overlap_count

        # 确保 remaining_count 不小于 0
        remaining_count = max(remaining_count, 0)
        
        # 如果 remaining_count > 0, 从剩余的区域中选择
        remaining_choices = []
        if remaining_count > 0 and len(remaining_choices) >= remaining_count:
            remaining_choices = random.sample(remaining_regions, remaining_count)
        
        # 如果还需要更多错误选项，则从所有区域中重新随机选择 (包括已正确区域)
        if len(remaining_choices) < remaining_count:
            extra_count = remaining_count - len(remaining_choices)
            extra_choices = random.sample(all_region_ids, extra_count)
            remaining_choices.extend(extra_choices)
        
        # 合并重叠区域和非重叠区域，作为错误选项
        wrong_choice = overlap_regions + remaining_choices
        
        # 检查是否生成的错误答案与正确答案完全相同，或是否与已有错误答案重复
        if set(wrong_choice) == set(correct_region_ids):  # 排除与正确答案相同的错误答案
            continue
        
        # 将错误答案转化为 frozenset 来确保其唯一性
        wrong_choices.add(frozenset(wrong_choice))  # frozenset 是不可变的集合，能够在 set 中保证唯一性
    
    # 将 frozenset 转换回 list，并打乱每个错误答案的顺序
    wrong_choices = [list(wrong_choice) for wrong_choice in wrong_choices]
    
    return wrong_choices


def get_locations(json_data, df):

# 存储 Location 列的结果
    locations = []
    # locations.append('locations')

    # 处理每个问题并将结果存入 'Location' 列
    for idx, row in df.iterrows():
        question = row['questions']
        
        # 提取物体名称
        objects = extract_objects_from_question(question)
        
        # 查找物体的位置
        location_result = []
        for obj in objects:
            details_list = find_object_details(obj, json_data)
            for details in details_list:
                # 按照指定格式输出
                location_result.append(f"{details['category_name']}: {details['dimensions']}, {details['region_id']}")
        
        # 将位置结果添加到 locations 列表
        if location_result:
            locations.append('; '.join(location_result))
        else:
            locations.append('No location found')
    return locations

def get_answers_options_label(json_data, df):

    answers = []
    options = []
    labels = []

    all_region_ids = get_all_region_ids(json_data)
    print('all_region_ids', all_region_ids)
    # locations.append('locations')

    # 处理每个问题并将结果存入 'Location' 列
    for idx, row in df.iterrows():
        question = row['questions']  # 假设问题列为 'questions'
        
        # 提取问题中的所有物体
        objects = extract_objects_from_question(question)
        
        if len(objects) == 1:
            # 单物体问题，提取物体的所有 region_id，并去重
            object_name = objects[0]
            correct_region_ids = find_object_region_ids(object_name, json_data)
            
            # 使用 set 去重后重新转换为 list
            unique_region_ids = list(set(correct_region_ids))
            answers.append(unique_region_ids)
            
            # 生成错误选项
            wrong_region_ids = generate_wrong_options(unique_region_ids, all_region_ids, num_wrong=3)
            
            # 随机打乱选项顺序，确保正确答案不会固定在A
            options_list = ['A', 'B', 'C', 'D']
            correct_option = random.choice(options_list)  # 随机选择正确答案的位置
            
            # 将正确选项放到随机选项位置
            option_dict = {correct_option: unique_region_ids}
            
            # 将错误选项放到其余位置
            wrong_options = iter(wrong_region_ids)
            for option in [opt for opt in options_list if opt != correct_option]:
                option_dict[option] = next(wrong_options)
            
            # 按照 A, B, C, D 顺序排列
            formatted_options = "; ".join([f"{key}. {option_dict[key]}" for key in options_list])
            options.append(formatted_options)
            
            # 将正确的选项加入标签列
            labels.append(correct_option)
        
        elif len(objects) == 2:
            # 多物体问题，提取两个物体的共有 region_id
            object_name_1 = objects[0]
            object_name_2 = objects[1]
            
            region_ids_1 = find_object_region_ids(object_name_1, json_data)
            region_ids_2 = find_object_region_ids(object_name_2, json_data)
            
            # 取两个物体共有的 region_id
            common_region_ids = list(set(region_ids_1) & set(region_ids_2))
            answers.append(common_region_ids)
            
            # 生成错误选项
            wrong_region_ids = generate_wrong_options(common_region_ids, all_region_ids, num_wrong=3)
            
            # 随机打乱选项顺序，确保正确答案不会固定在A
            options_list = ['A', 'B', 'C', 'D']
            correct_option = random.choice(options_list)  # 随机选择正确答案的位置
            
            # 将正确选项放到随机选项位置
            option_dict = {correct_option: common_region_ids}
            
            # 将错误选项放到其余位置
            wrong_options = iter(wrong_region_ids)
            for option in [opt for opt in options_list if opt != correct_option]:
                option_dict[option] = next(wrong_options)
            
            # 按照 A, B, C, D 顺序排列
            formatted_options = "; ".join([f"{key}. {option_dict[key]}" for key in options_list])
            options.append(formatted_options)
            
            # 将正确的选项加入标签列
            labels.append(correct_option)
        
        else:
            # 对于无法识别的情况，给出空列表
            answers.append([])
            options.append("")
            labels.append("")

    return answers, options, labels 

def format_question(row):
    # 从 options 列提取选项
    options = row['options'].split(';')  # 假设选项是以分号分隔的
    # 格式化问题和选项
    formatted_question = f"{row['questions']} A) {options[0].strip()} B) {options[1].strip()} C) {options[2].strip()} D) {options[3].strip()}. Answer:"
    return formatted_question


# 调用主函数
excel_path = 'question_HkseAnWCgqk_2.csv'  # 替换为你的Excel文件路径
json_path = 'HkseAnWCgqk_objects.json'  # 替换为你的JSON文件路径

output_path = 'question_answer_HkseAnWCgqk_all.csv'

df = pd.read_csv(excel_path)
with open(json_path, 'r') as f:
    json_data = json.load(f)


locations = get_locations(json_data, df)
answers, options, labels = get_answers_options_label(json_data, df)

# 存储 Location 列的结果

df['locations'] = locations




# 将结果存储到现有的 'answers' 列中
df['answers'] = answers
df['options'] = options
df['label'] = labels

# 直接保存到原始的 Excel 文件
df.to_csv(output_path, index=False)

print(f"Answers and options have been added to the original Excel file: {excel_path}")

df = pd.read_csv(output_path)
# 创建新的列 'question_formatted'，并填入格式化后的内容
df['question_formatted'] = df.apply(format_question, axis=1)

df.to_csv(output_path, index=False)

print("处理完成，文件已保存为")