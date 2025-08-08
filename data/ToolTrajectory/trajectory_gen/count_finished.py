import os

# 你的5个jsonlines文件路径
files = [
    "data/ToolTrajectory/trajectory_gen/status/output/status.jsonl",
    "data/ToolTrajectory/trajectory_gen/relationship/output/relationship.jsonl",
    "data/ToolTrajectory/trajectory_gen/location/special/output/special.jsonl",
    "data/ToolTrajectory/trajectory_gen/attribute/special/output/special.jsonl",
]

total_lines = 0

for file_path in files:
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
        print(f"{file_path}: {line_count} 行")
        total_lines += line_count

print(f"总行数: {total_lines}")
