cd /home/zml/algorithm/ReactEQA/data/ToolTrajectory/

echo "RUN trajectory_gen/post_process_observation.py"
python trajectory_gen/post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/status/output/status.jsonl' --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/status/output/status.jsonl'

echo "RUN evaluation/rule_evaluation.py"
python evaluation/rule_evaluation.py --file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/status/output/status.jsonl' --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/status/wrong_id.json'

echo "RUN evaluation/auto_evaluation.py"
python evaluation/auto_evaluation.py --function "answer_consist" --file "/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/status/output/status.jsonl" --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/status/wrong_id.json'

echo "RUN status_reproduct.py"
cd trajectory_gen/status
python status_reproduct.py