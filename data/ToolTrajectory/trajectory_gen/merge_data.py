import json
import jsonlines
import os

# 多个jsonlines文件合并为一个json文件
def merge_jsonlines_to_json(jsonlines_files, output_json_file):
    merged_data = []
    for jsonlines_file in jsonlines_files:
        with jsonlines.open(jsonlines_file, 'r') as reader:
            for item in reader:
                merged_data.append(item)
        
        print(f"Processing {jsonlines_file}...", len(merged_data))
    
    with open(output_json_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

if __name__ == "__main__":
    # Example usage
    jsonlines_files = [
        'data/ToolTrajectory/trajectory_gen/status/output/status_reproduct.jsonl',
        'data/ToolTrajectory/trajectory_gen/relationship/output/relationship_reproduct.jsonl',
        'data/ToolTrajectory/trajectory_gen/location/location/output/location_one_reproduct.jsonl',
        'data/ToolTrajectory/trajectory_gen/location/location/output/location_two_reproduct.jsonl',
        'data/ToolTrajectory/trajectory_gen/location/special/output/special_reproduct.jsonl',
        'data/ToolTrajectory/trajectory_gen/distance/output/distance_reproduct.jsonl',
        'data/ToolTrajectory/trajectory_gen/counting/output/counting_reproduct.jsonl',
        'data/ToolTrajectory/trajectory_gen/attribute/color/output/color_reproduct.jsonl',
        'data/ToolTrajectory/trajectory_gen/attribute/size/output/size_reproduct.jsonl',
        'data/ToolTrajectory/trajectory_gen/attribute/special/output/special_reproduct.jsonl',
    ]
    output_json_file = 'data/ToolTrajectory/trajectory_gen/trajectory.json'
    
    merge_jsonlines_to_json(jsonlines_files, output_json_file)
    print(f"Merged data saved to {output_json_file}")