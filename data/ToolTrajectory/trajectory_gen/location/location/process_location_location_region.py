import os
import json

# 路径配置
input_json_path = 'data/location-location_revise_end_2.json'  # 你的原始json路径
hm3d_root = '/data/zml/datasets/EmbodiedQA/HM3D'

data_new = []

data_remove = []
# 读取原始JSON
with open(input_json_path, 'r') as f:
    data = json.load(f)
    
proposal_choice = ["A", "B", "C", "D"]
all_data_number = 0
success_data_number = 0
save_data_number = 0
stat_answer = {}
# stat_answer["1"] = 0
# stat_answer["2"] = 0
# stat_answer["3"] = 0
# stat_answer["4"] = 0
# stat_answer["5"] = 0

for i in range(20):
    stat_answer[str(i)] = 0

# 遍历每个条目
for item in data:
    scene_id = item.get('scene')
    proposals = item.get('proposals', [])
    all_data_number = all_data_number +1 

    
    if not scene_id or not proposals:
        continue
    # 获取 region.json 文件路径
    folder_path = os.path.join(hm3d_root, scene_id)
    if not os.path.isdir(folder_path):
        print(f"[警告] 找不到场景文件夹: {scene_id}")
        continue

    # 找出 region.json 文件（如 HkseAnWCgqk.region.json）
    try:
        region_json_file = next(
            f for f in os.listdir(folder_path)
            if f.endswith('.region.json')
        )
    except StopIteration:
        print(f"[警告] 找不到 region.json 文件: {scene_id}")
        continue

    region_json_path = os.path.join(folder_path, region_json_file)

    # 加载 region_id -> region_name 映射
    with open(region_json_path, 'r') as rf:
        region_data = json.load(rf)

    # 如果是 list 格式（如 region 列表）
    if isinstance(region_data, list):
        id_name_map = {}
        for region in region_data:
            region_id = region.get("region_id")  # e.g. "region_7"
            region_name = region.get("region_name")
            if region_id and region_name and region_id.startswith("region_"):
                numeric_id = region_id.replace("region_", "")  # 提取出 "7"
                id_name_map[numeric_id] = region_name
    else:
        print(f"[警告] region.json 文件内容不是列表: {region_json_path}")
        continue

    # 替换 proposals 中的 ID 为 region_name
    new_proposals = []
    success = True
    for group in proposals:
        if isinstance(group, str):
            ids = [x.strip() for x in group.split(',')]
            names = [id_name_map.get(x) for x in ids]
            if None in names:
                success = False
                break  # 跳过这个 group 
            new_proposals.append(', '.join(names))
            
        else:
            success = False
            print('---------------')
            break
            # new_proposals.append(group)  # 保留原样
    
    if success == False:
        continue
    
    print('proposals', proposals, 'new_proposals', new_proposals)

    # 替换原来的 proposals
    item['proposals'] = new_proposals

    answer = new_proposals[proposal_choice.index(item["answer"][0].upper())]

    answer_list = [room.strip() for room in answer.split(',')]

    stat_answer[str(len(answer_list))] = stat_answer[str(len(answer_list))] + 1
    success_data_number = success_data_number + 1
    if len(answer_list)>2:
        continue
    
    related_objects_revise = item["related_objects"]

    for obj_r in related_objects_revise:
        region_id_obj = obj_r["region_id"]
        region_name_obj = id_name_map.get(str(region_id_obj)) 
        print('id_name_map', id_name_map)
        print('region_id_obj',region_id_obj,'region_name_obj', region_name_obj)
        obj_r["region_id"] = region_name_obj

    data_new.append(item)
    # item['proposals'] = proposals
    # data_remove.append(item)
    save_data_number = save_data_number + 1
# 可选：保存到新文件


output_path = 'data/location-location_revise_end_3.json'
with open(output_path, 'w') as out_f:
    json.dump(data_new, out_f, indent=2)

# output_path = 'data/location-location_remove.json'
# with open(output_path, 'w') as out_f:
#     json.dump(data_remove, out_f, indent=2)

print(f"[完成] 已处理并保存为：{output_path}")

print('success_data_number', success_data_number, 'all_data_number', all_data_number, 'save_data_number', save_data_number) 

print(stat_answer)






