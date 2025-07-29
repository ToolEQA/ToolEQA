import json

# 原始文件路径
input_file = "trajectory-wo_react.json"

# 输出文件路径
output_file = "color_attributes.json"

# 读取 JSON 文件
with open(input_file, "r") as f:
    data = json.load(f)

# 过滤出 attribute 为 color 的项
color_items = [item for item in data if item.get("question_type") == "attribute-color"]

# 写入新的 JSON 文件
with open(output_file, "w") as f:
    json.dump(color_items, f, indent=4)

print(f"已保存 {len(color_items)} 个 attribute=color 的项到 {output_file}")
