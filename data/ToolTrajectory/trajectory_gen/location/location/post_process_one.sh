cd /home/zml/algorithm/ReactEQA/data/ToolTrajectory/

echo "RUN trajectory_gen/post_process_observation.py"
python trajectory_gen/post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/location/location/output/location_one.jsonl' --output_file  '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/location/location/output/location_one.jsonl'

echo "RUN evaluation/rule_evaluation.py"
python evaluation/rule_evaluation.py --file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/location/location/output/location_one.jsonl' --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/location/location/wrong_id_one.json'

echo "RUN evaluation/auto_evaluation.py"
python evaluation/auto_evaluation.py --function "answer_consist" --file "/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/location/location/output/location_one.jsonl" --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/location/location/wrong_id_one.json'

echo "RUN location_reproduct.py"
cd trajectory_gen/location/location
python location_location_one_reproduct.py