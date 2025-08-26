import json
with open("/home/zml/data/EQA-Traj-0720/test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

output_file = "output.json"


# 遍历删除 action 字段
for item in data:
    if "trajectory" in item:
        for step in item["trajectory"]:
            step.pop("action", None)  # 如果有 action 就删掉
            step.pop("position", None)  # 如果有 action 就删掉
            step.pop("rotation", None)  # 如果有 action 就删掉


# 保存到新文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"已生成新文件: {output_file}")