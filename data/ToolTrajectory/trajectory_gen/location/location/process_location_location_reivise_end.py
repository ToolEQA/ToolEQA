import os
import json

# 配置路径
input_json_path = 'data/location-location_region_revise.json'  # 替换为你的输入路径
hm3d_root = '/data/zml/datasets/EmbodiedQA/HM3D'

# 读取主 JSON 数据
with open(input_json_path, 'r') as f:
    data = json.load(f)

proposal_choice = ["A", "B", "C", "D"]

# 处理每一项
for entry_idx, entry in enumerate(data):
    scene = entry.get("scene")
    related_objects = entry.get("related_objects", [])

    proposals = entry["proposals"]
    answer = proposals[proposal_choice.index(entry["answer"][0].upper())]

    answer_list = [room.strip() for room in answer.split(',')]

    if not scene or not related_objects:
        continue

    scene_folder = os.path.join(hm3d_root, scene)
    if not os.path.isdir(scene_folder):
        print(f"[警告] 找不到文件夹: {scene_folder}")
        continue

    # 构建 objects.cleaned.json 路径
    try:
        scene_id_part = scene.split('-')[1]
        object_file = os.path.join(scene_folder, f"{scene_id_part}.objects.cleaned.json")
    except IndexError:
        print(f"[错误] scene 格式不正确: {scene}")
        continue

    if not os.path.exists(object_file):
        print(f"[缺失] 找不到对象文件: {object_file}")
        continue

    # 读取 cleaned 对象数据
    with open(object_file, 'r') as f:
        scene_objects = json.load(f)

    # 构建 object_id 到 region_id 映射
    object_region_map = {
        obj["object_id"]: obj["region_id"]
        for obj in scene_objects if "object_id" in obj and "region_id" in obj
    }

    # 添加 region_id 到 related_objects，并去重
    seen = set()
    new_related_objects = []
    removed_indices = []

    for idx, obj in enumerate(related_objects):
        obj_id = obj.get("id")
        name = obj.get("name")
        region_id = obj.get("region_id")

        if region_id not in answer_list:
            removed_indices.append(idx)
        else:
            new_related_objects.append(obj)




        

    # 更新数据结构
    entry["related_objects"] = new_related_objects
    # entry["removed_indices"] = removed_indices

        # ========= 根据 removed_indices 清理 trajectory =========
    if "trajectory" in entry and isinstance(entry["trajectory"], list):
        original_len = len(entry["trajectory"])
        cleaned_trajectory = []
        for traj_item in entry["trajectory"]:
            step = traj_item.get("step", "")
            # 只保留 step 不以 removed index 开头的项
            if not any(step.startswith(f"{idx}-") for idx in removed_indices):
                cleaned_trajectory.append(traj_item)

        # 更新 trajectory
        entry["trajectory"] = cleaned_trajectory
        print(f"[清理] scene {scene}: 删除了 {original_len - len(cleaned_trajectory)} 条 trajectory step")
        # 重新编号 trajectory 中的 step 主编号，使其连续
        # 输入例子: step = "3-0" → 我们要把 "3" 改成连续的编号

        # 第一步：收集所有主编号出现的顺序
        old_main_ids = []
        for traj_item in cleaned_trajectory:
            step = traj_item.get("step", "")
            parts = step.split('-')
            if len(parts) != 2:
                continue
            main_id = int(parts[0])
            if main_id not in old_main_ids:
                old_main_ids.append(main_id)

        # 建立 old → new 主编号映射
        main_id_map = {old: new for new, old in enumerate(old_main_ids)}

        # 第二步：替换 step
        for traj_item in cleaned_trajectory:
            step = traj_item.get("step", "")
            parts = step.split('-')
            if len(parts) == 2:
                old_main = int(parts[0])
                sub_id = parts[1]
                new_main = main_id_map.get(old_main, old_main)
                traj_item["step"] = f"{new_main}-{sub_id}"
            else:
                print(f"[警告] step 格式错误: {step}")

        # 最后更新 entry
        entry["trajectory"] = cleaned_trajectory


    print(f"[处理] scene {scene}: 移除 {len(removed_indices)} 个重复对象")

# 可选：保存处理后的 JSON
output_path = 'data/location-location_revise_end.json'
with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n✅ 全部处理完成，结果已保存至：{output_path}")
