import os
import pickle
import json
from tqdm import tqdm

def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def save_pkl(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

# 主程序
if __name__ == "__main__":
    root = "data/HM3D"
    dirs = os.listdir(root)
    dirs.sort()

    floor_info_file = "data/HM3D/scene_floor_info.json"
    with open(floor_info_file, "r") as f:
        floor_info = json.load(f)

    for dir in tqdm(dirs):
        path = os.path.join(root, dir)
        if not os.path.isdir(path):
            continue
        number, scene_id = dir.split("-")

        objects_pkl = os.path.join(path, scene_id + ".objects.pkl")
        output_pkl = os.path.join(path, scene_id + ".objects.extra.pkl")
        if not os.path.exists(objects_pkl):
            continue
        floor_list = floor_info[scene_id]
        objects_data = load_pkl(objects_pkl)
        for obj in objects_data["objects"]:
            height = obj["bbox"]["center"][2]
            for k, v in floor_list.items():
                if height >= v[0] and height <= v[1]:
                    obj["floor_id"] = k
            assert "floor_id" in obj.keys(), "对象没在任何楼层里"

        save_pkl(output_pkl, objects_data)