import os
import pandas as pd
import csv
import ast
import re

class Filter():
    def __init__(self, files):
        self.files = files
        self.seen_counting_keys = set()

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
        if len(categories) != len(set(categories)) and str(data.get('label', '')).strip() != 'counting':
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

    def check(self, data):
        checked = []
        checked.append(self.check_obj(data))
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


    def filtering(self):
        for file in self.files:
            question_csv = pd.read_csv(file)
            filtered_data = []
            self.seen_counting_keys = set()
            for index, row in question_csv.iterrows():
                data = dict(row)
                if self.check(data):
                    filtered_data.append(data) 
            # self.write_csv(file + ".filtered", filtered_data)
            print(index+1, len(filtered_data))

if __name__=="__main__":
    files_path = [
        "data/ToolTrajectory/questions/attribute/color.csv",
        "data/ToolTrajectory/questions/attribute/size.csv",
        "data/ToolTrajectory/questions/attribute/special.csv",
        "data/ToolTrajectory/questions/counting/counting.csv",
        "data/ToolTrajectory/questions/distance/distance.csv",
        "data/ToolTrajectory/questions/location/location.csv",
        "data/ToolTrajectory/questions/location/special.csv",
        "data/ToolTrajectory/questions/relationship/relationship.csv",
        "data/ToolTrajectory/questions/status/status.csv"
    ]
    filter = Filter(files_path)
    filter.filtering()