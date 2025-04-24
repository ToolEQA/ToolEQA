import pandas as pd
import json
import re
import random
import jieba.posseg as pseg
import nltk
from nltk.corpus import wordnet
lemmatizer = nltk.WordNetLemmatizer()

# 读取CSV文件
def read_csv(file_path):
    return pd.read_csv(file_path)

# 读取JSON文件
def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# def remove_plural_s(object_name):
#     if object_name.endswith('s'):
#         return object_name[:-1]  # 去掉最后的s
#     return object_name

def remove_plural_s(object_name):
    if object_name.endswith('es'):  # 去掉以"es"结尾的复数
        return object_name[:-2]  # 去掉"es"
    elif object_name.endswith('s'):  # 去掉以"s"结尾的复数
        return object_name[:-1]  # 去掉"s"
    return object_name

def to_singular(word):
    # 使用 lemmatizer，将复数名词转换为单数
    words = word.split()  # 拆分多词组合名词
    # 对最后一个词进行复数转换
    last_word = words[-1]  # 获取最后一个词
    singular_last_word = lemmatizer.lemmatize(last_word, pos=wordnet.NOUN)   # 处理复数到单数
    words[-1] = singular_last_word  # 更新最后一个词
    return ' '.join(words)  # 将单词重新组合成短语
    # return lemmatizer.lemmatize(word, pos=wordnet.NOUN) 




def extract_info(sentence):
    # 定义正则表达式模式
    # single_object_single_region = r"How many (\w+) are in region (\d+)\?"
    # single_object_multiple_regions = r"How many (\w+) are in region (\d+) and region (\d+)\?"
    # multiple_objects_single_region = r"How many (\w+ and \w+) are in region (\d+)\?"

    single_object_single_region = r"How many (.+?) are in region (\d+)\?"
    single_object_multiple_regions = r"How many (.+?) are in region (\d+) and region (\d+)\?"
    # multiple_objects_single_region = r"How many (.+?) are in region (\d+)\?"
    single_object_scene = r"How many (.+?) are in the scene\?"

    # 单物体单区域
    match1 = re.match(single_object_single_region, sentence)
    if match1:
        object_name = match1.group(1).strip()
        region = match1.group(2)
        # 如果物体名称中包含 "and"，就拆开成多个物体
        objects = [obj.strip() for obj in object_name.split(" and ")]
        return {"objects": objects, "regions": [region]}

    # 单物体多区域
    match2 = re.match(single_object_multiple_regions, sentence)
    if match2:
        object_name = match2.group(1).strip()
        regions = [match2.group(2), match2.group(3)]
        # 如果物体名称中包含 "and"，就拆开成多个物体
        objects = [obj.strip() for obj in object_name.split(" and ")]
        return {"objects": objects, "regions": regions}


    match3 = re.match(single_object_scene, sentence)
    if match3:
        object_name = match3.group(1).strip()
        # region = match3.group(2)
        # 如果物体名称中包含 "and"，就拆开成多个物体
        objects = [obj.strip() for obj in object_name.split(" and ")]
        return {"objects": objects, "regions": ['scene']}

    # # 多物体单区域
    # match3 = re.match(multiple_objects_single_region, sentence)
    # if match3:
    #     object_name = match3.group(1).strip()
    #     region = match3.group(2)
    #     # 如果物体名称中包含 "and"，就拆开成多个物体
    #     objects = [obj.strip() for obj in object_name.split(" and ")]
    #     return {"objects": objects, "regions": [region]}

    # 如果没有匹配到
    return None



# 从JSON数据中查找物体对应的信息
def get_object_details(objects, regions, json_data):
    result = []
    # print('objects_original', objects)
    for region in regions:
        # print('region', region)
        if region == 'scene':
            for obj in objects:
                # 查找物体名称对应的类别，位置等信息
                print('objects_before', obj)
                obj_singular = to_singular(obj)  # 去复数s
                print('objects_before', obj_singular)
                for item in json_data:
                    if obj_singular.lower() == item["category_name"].lower():
                        # print('yes yes yes yes yes yes yes')
                        category_name = item["category_name"]
                        position = item["center"]
                        region_id = item["region_id"]
                        result.append(f"{category_name}: {position}, {region_id}")
                    # else:
                    #     print('no', regions, objects)
        else:
            for obj in objects:
                # print('objects_after', obj)
                print('objects_before', obj)
                obj_singular = to_singular(obj)  # 去复数s
                print('objects_before', obj_singular)
                # 查找物体名称对应的类别，位置等信息
                for item in json_data:
                    if str(item["region_id"]) == str(region) and obj_singular.lower() == item["category_name"].lower():
                        # print('yes yes yes yes yes yes yes')
                        category_name = item["category_name"]
                        position = item["center"]
                        region_id = item["region_id"]
                        result.append(f"{category_name}: {position}, {region_id}")
                        # else:
                        #     print('no', regions, objects)
    return  len(result), "; ".join(result)

# 主函数
def process_questions_and_update_locations(excel_path, json_path, output_path):
    # 读取Excel和JSON文件
    df = read_csv(excel_path)
    json_data = read_json(json_path)

    # 对每个问题进行处理
    locations = []
    countings_obj = []

    for question in df['questions']:
        print(question)
        extracted_info = extract_info(question)
        if extracted_info:
            objects = extracted_info['objects']
            regions = extracted_info['regions']
            # print(question)
            # print(objects, regions)
            # 获取每个物体的详细信息

            countings, location_info = get_object_details(objects, regions, json_data)
            # print(location_info)
            locations.append(location_info)
            countings_obj.append(countings)
        else:
            locations.append("")  # 如果无法提取信息，则填充空值
            countings_obj.append(0)
    df['locations'] = locations
    df['answers'] = countings_obj
    df.to_csv(output_path, index=False)
    print(f"Processed data has been saved to {output_path}")



def generate_options_and_label(answer):
    # 确保答案是一个整数
    correct_answer = int(answer)
    # print('correct_answer', correct_answer)
    # 生成四个选项，其中包括正确答案和三个不同的选项
    options = set()  # 使用集合来保证唯一性
    options.add(correct_answer)
    # 我们通过对正确答案加减1,2,3等生成其他选项
    while len(options) < 4:
        delta = random.choice([-3, -2, -1, 1, 2, 3])  # 随机加减1,2,3
        option = correct_answer + delta
        if option > 0:  # 确保选项是正整数
            options.add(option)
        # print(options)
    
    # 将选项转化为列表并排序
    options = list(options)
    
    # 随机打乱选项，确保正确答案不会总是出现在同一个位置
    random.shuffle(options)
    
    # 构建选项的格式
    options_str = f"A. {options[0]}; B. {options[1]}; C. {options[2]}; D. {options[3]}"
    
    # 确定正确答案的标签
    label = ['A', 'B', 'C', 'D'][options.index(correct_answer)]
    
    return options_str, label

def process_answers_and_generate_options(csv_path, output_path):
    # 读取CSV文件
    df = read_csv(csv_path)

    # 对每个问题的答案生成选项和标签
    options = []
    labels = []
    for answer in df['answers']:
        option_str, label = generate_options_and_label(answer)
        options.append(option_str)
        labels.append(label)

    # 将生成的选项和标签添加到DataFrame中
    df['options'] = options
    df['label'] = labels
    
    # 将更新后的DataFrame保存为CSV文件
    df.to_csv(output_path, index=False)
    print(f"Processed data has been saved to {output_path}")

def format_question(row):
    # 从 options 列提取选项
    options = row['options'].split(';')  # 假设选项是以分号分隔的
    # 格式化问题和选项
    formatted_question = f"{row['questions']} A) {options[0].strip()} B) {options[1].strip()} C) {options[2].strip()} D) {options[3].strip()}. Answer:"
    return formatted_question



excel_path = 'question_HkseAnWCgqk_counting_answer.csv'  # 替换为你的Excel文件路径
json_path = 'HkseAnWCgqk_objects.json'  # 替换为你的JSON文件路径
# find_object_location(excel_path, json_path)
# output_path = 'question_HkseAnWCgqk_counting_answer.csv'  # 输出CSV文件路径

process_questions_and_update_locations(excel_path, json_path, excel_path)
process_answers_and_generate_options(excel_path, excel_path)

excel_file = 'question_HkseAnWCgqk_counting_answer.csv'
df = pd.read_csv(excel_file)
# 创建新的列 'question_formatted'，并填入格式化后的内容
df['question_formatted'] = df.apply(format_question, axis=1)
df.to_csv(excel_file, index=False)