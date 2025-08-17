import json

data_list = [
    "data/ToolTrajectory/trajectory_gen/attribute/color/output/color_wrong.json",
    "data/ToolTrajectory/trajectory_gen/attribute/color/output/wrong_answer.json"
]

output_data = {}

for data in data_list:
    with open(data) as f:
        data = json.load(f)
    for key in data.keys():
        if key not in output_data:
            output_data[key] = data[key]
        else:
            step_data = list(set(data[key] + output_data[key]))
            output_data[key] = step_data

with open("data/ToolTrajectory/trajectory_gen/attribute/color/output/all_wrong.json", "w") as f:
    json.dump(output_data, f, indent=4)