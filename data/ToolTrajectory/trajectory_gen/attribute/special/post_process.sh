cd /home/zml/algorithm/ReactEQA/data/ToolTrajectory/

echo "RUN trajectory_gen/post_process_observation.py"
python trajectory_gen/post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/output/special.jsonl' --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/output/special.jsonl'

echo "RUN evaluation/rule_evaluation.py"
python evaluation/rule_evaluation.py --file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/output/special.jsonl' --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/wrong_id.json'

# echo "RUN evaluation/auto_evaluation.py"
# python evaluation/auto_evaluation.py --function "answer_consist" --file "/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/output/special.jsonl" --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/wrong_id.json'

echo "RUN special_reproduct.py"
cd trajectory_gen/attribute/special
python special_reproduct.py

# rm /home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/output/special.jsonl
# mv /home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/output/special_reproduct.jsonl /home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/special/output/special.jsonl