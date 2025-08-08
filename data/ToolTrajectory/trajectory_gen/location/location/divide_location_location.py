

import json

# 设置你的JSON路径
json_path = 'data/location-location_revise_final.json'  # 替换为你的真实路径

# 加载JSON数据
with open(json_path, 'r') as f:
    data = json.load(f)

all_object = 0
count_greater_than_3 = 0
proposal_choice = ["A", "B", "C", "D"]
wrong_item = 0

data_one = []

data_two = []


for idx, item in enumerate(data):
    objects = item.get('related_objects', [])
    length = len(objects)
    # print(f"Item {idx}: objects 长度 = {length}")

    proposals = item["proposals"]
    answer = proposals[proposal_choice.index(item["answer"][0].upper())]

    answer_list = [room.strip() for room in answer.split(',')]

    region_objects = []
    object_name = []
    for obj_i in objects:
        region_objects.append(str(obj_i["region_id"]).strip())
        object_name.append(obj_i["name"])
    
    if set(answer_list) != set(region_objects):
        wrong_item = wrong_item +1 
        print(item["sample_id"], set(answer_list), set(region_objects))

    if len(set(object_name))==1:
        data_one.append(item)
    else:
        data_two.append(item)

    all_object = all_object + 1

output_path = 'data/location-location_oneobject.json'
with open(output_path, 'w') as out_f:
    json.dump(data_one, out_f, indent=2)

output_path = 'data/location-location_twoobject.json'
with open(output_path, 'w') as out_f:
    json.dump(data_two, out_f, indent=2)

print('all_object', all_object)
print(f"\n共有 {count_greater_than_3} 个项的 'objects' 长度大于 3")
print('wrong_item', wrong_item)

print('data_one', len(data_one), 'data_two', len(data_two))