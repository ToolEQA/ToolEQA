python post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/size/output/size.jsonl' --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/size/output/size_revise.jsonl' >& post_attribute_size.log

python post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/output/special.jsonl' --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/output/special_revise.jsonl' >& post_attibute_special.log

python post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/status/output/status.jsonl' --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/status/output/status_revise.jsonl' >& post_status.log

python post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/relationship/output/relationship.jsonl' --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/relationship/output/relationship_revise.jsonl' >& post_relationship.log

python post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/counting/output/counting.jsonl' --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/counting/output/counting_revise.jsonl' >& post_counting.log

python post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/location/location/output/location_one.jsonl' --output_file  '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/location/location/output/location_one_revise.jsonl' >& post_location_location_one.log

python post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/location/location/output/location_two.jsonl' --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/location/location/output/location_two_revise.jsonl' >& post_location_location_two.log

python post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/location/special/output/special.jsonl' --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/location/special/output/special_revise.jsonl' >& post_location_special.log

python post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/distance/output/distance.jsonl' --output_file  '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/distance/output/distance_revise.jsonl' >& post_distance.log

python post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/color/output/color.jsonl' --output_file  '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/color/output/color_revise.jsonl' >& log/post_color.log
