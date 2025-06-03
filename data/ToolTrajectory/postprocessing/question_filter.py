import os
import pandas as pd
import csv
import ast
import re

class Filter():
    def __init__(self, files):
        self.files = files
        self.seen_counting_keys = set()
        self.cur_type = ""

    def check_refering(self, data):
        # 删除问题中包含this image、this scene，this object、that area这类指代不明的词汇
        conditions = [
            "this image",
            "this scene",
            "this object",
            "that area",
            "this area"
        ]
        question = data["question"]
        for condition in conditions:
            if condition in question:
                return False
            
        return True
    
    def check_empty(self, data):
        if any(pd.isna(data[key]) or str(data[key]).strip() == "" for key in data.keys()):
            return False
        return True
        
    def check_choice(self, data):
        try:
            choices = ast.literal_eval(data['choices']) if isinstance(data['choices'], str) else data['choices']
            if not isinstance(choices, list) or any(not str(c).strip() for c in choices) or len(choices) != len(set(choices)):
                return False
        except Exception:
            return False
        return True
        
    def check_unique_cate(self, data):
        loc_str = str(data['locations'])
        categories = re.findall(r'\b([a-zA-Z_]+) \(.*?\d+\)', loc_str)
        if len(categories) != len(set(categories)) and str(data.get('label', '')).strip() not in ['counting', 'location']:
            return False
        return True
    
    def check_repeat(self, data):
        scene_id = data['scene_id']
        loc_str = str(data['locations'])
        categories = re.findall(r'\b([a-zA-Z_]+) \(.*?\d+\)', loc_str)
        object_types = tuple(sorted(set(categories)))
        key = (scene_id, object_types)
        if key in self.seen_counting_keys:
            return False
        self.seen_counting_keys.add(key)
        return True

    def check_answer(self, data):
        answer = data['answer']
        if not isinstance(answer, str):
            return False
        
        if answer.strip() in ["", "[]"]:
            return False
        return True

    def check_locations(self, data):
        loc_str = data['locations']
        if loc_str in [""]:
            return False
        
        return True

    def check_count(self, data):
        if self.cur_type in ["location_location"]:
            return True
        
        loc_str = data['locations']
        if not isinstance(loc_str, str):
            return False
        loc_list = loc_str.split(";")
        if self.cur_type in ["counting_counting"]:
            if len(loc_list) > 5:
                return False
        else:
            if len(loc_list) > 3:
                return False
        return True

    def check(self, data):
        checked = []
        checked.append(self.check_count(data))
        checked.append(self.check_locations(data))
        checked.append(self.check_answer(data))
        checked.append(self.check_refering(data))
        checked.append(self.check_empty(data))
        checked.append(self.check_choice(data))
        checked.append(self.check_unique_cate(data))
        checked.append(self.check_repeat(data))
        return len(checked) == sum(checked)

    def write_csv(self, file_path, data):
        if len(data) > 0:
            title = data[0].keys()
        else:
            return

        if os.path.exists(file_path):
            with open(file_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=title)
                writer.writerows(data)
        else:
            with open(file_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=title)
                writer.writeheader()
                writer.writerows(data)

    def save_csv(self, root, data):
        types = self.cur_type.split("_")
        if not os.path.exists(os.path.join(root, types[0])):
            os.mkdir(os.path.join(root, types[0]))
        path = os.path.join(root, types[0], types[1] + ".csv")
        self.write_csv(path, data)

    def filtering(self):
        org_data = 0
        all_data = 0
        for file in self.files:
            question_csv = pd.read_csv(file)
            filtered_data = []
            self.seen_counting_keys = set()
            self.cur_type = file.split("/")[-2] + "_" + file.split("/")[-1][:-4]

            for index, row in question_csv.iterrows():
                data = dict(row)
                if self.check(data):
                    filtered_data.append(data) 
            
            # 保存筛选后的样本文件
            # self.write_csv(file + ".filtered", filtered_data)
            self.save_csv("data/ToolTrajectory/questions/final_question", filtered_data)

            all_data += len(filtered_data)
            org_data += index+1
            print(file, index+1, len(filtered_data))
        print(f"原始{org_data}条样本，过滤后还剩{all_data}条样本")

if __name__=="__main__":
    files_path = [
        "data/ToolTrajectory/questions/raw_question/attribute/color.csv",
        "data/ToolTrajectory/questions/raw_question/attribute/size.csv",
        "data/ToolTrajectory/questions/raw_question/attribute/special.csv",
        "data/ToolTrajectory/questions/raw_question/counting/counting.csv",
        "data/ToolTrajectory/questions/raw_question/distance/distance.csv",
        "data/ToolTrajectory/questions/raw_question/location/location.csv",
        "data/ToolTrajectory/questions/raw_question/location/special.csv",
        "data/ToolTrajectory/questions/raw_question/relationship/relationship.csv",
        "data/ToolTrajectory/questions/raw_question/status/status.csv"
    ]
    filter = Filter(files_path)
    filter.filtering()