cd /home/zml/algorithm/ReactEQA/data/ToolTrajectory/

echo "RUN trajectory_gen/post_process_observation.py"
python trajectory_gen/post_process_observation.py --input_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/color/output/color.jsonl' --output_file  '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/color/output/color.jsonl'

echo "RUN evaluation/rule_evaluation.py"
python evaluation/rule_evaluation.py --file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/color/output/color.jsonl' --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/color/wrong_id.json'

echo "RUN evaluation/auto_evaluation.py"
python evaluation/auto_evaluation.py --function "answer_consist" --file "/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/color/output/color.jsonl" --output_file '/home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/color/wrong_id.json'

echo "RUN color_reproduct.py"
cd trajectory_gen/attribute/color
python color_reproduct.py

# rm /home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/color/output/color.jsonl
# mv /home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/color/output/color_reproduct.jsonl /home/zml/algorithm/ReactEQA/data/ToolTrajectory/trajectory_gen/attribute/color/output/color.jsonl