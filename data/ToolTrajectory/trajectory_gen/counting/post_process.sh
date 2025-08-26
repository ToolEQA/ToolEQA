cd /home/zml/algorithm/ReactEQA/data/ToolTrajectory/

echo "RUN trajectory_gen/post_process_observation.py"
python trajectory_gen/post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/counting/output/counting.jsonl' --output_file  '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/counting/output/counting.jsonl'

echo "RUN evaluation/rule_evaluation.py"
python evaluation/rule_evaluation.py --file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/counting/output/counting.jsonl' --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/counting/wrong_id.json'

# echo "RUN evaluation/auto_evaluation.py"
# python evaluation/auto_evaluation.py --function "answer_consist" --file "/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/counting/output/counting.jsonl" --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/counting/wrong_id.json'

echo "RUN counting_reproduct.py"
cd trajectory_gen/counting
python counting_reproduct.py