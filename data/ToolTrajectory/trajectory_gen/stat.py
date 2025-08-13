import json
import jsonlines
import os

# 多个jsonlines文件合并为一个json文件
def stat(jsonlines_files):
    all_count = 0
    for jsonlines_file in jsonlines_files:
        with open(jsonlines_file, 'r') as f:
            _data = json.load(f)
        print(f"Processing {jsonlines_file}...", len(_data))
        all_count += len(_data)
    print(f"Total count: {all_count}")

if __name__ == "__main__":
    # Example usage
    jsonlines_files = [
        'data/ToolTrajectory/trajectory_gen/attribute/color/data/attribute-color.json',
        'data/ToolTrajectory/trajectory_gen/attribute/size/data/attribute-size.json',
        'data/ToolTrajectory/trajectory_gen/attribute/special/data/attribute-special.json',
        'data/ToolTrajectory/trajectory_gen/counting/data/counting-counting.json',
        'data/ToolTrajectory/trajectory_gen/distance/data/distance-distance.json',
        'data/ToolTrajectory/trajectory_gen/location/location/data/location-location.json',
        'data/ToolTrajectory/trajectory_gen/location/special/data/location-special.json',
        'data/ToolTrajectory/trajectory_gen/relationship/data/relationship-relationship.json',
        'data/ToolTrajectory/trajectory_gen/status/data/status-status.json',
    ]
    
    stat(jsonlines_files)