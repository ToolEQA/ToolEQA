TMP_FILE=data/ToolTrajectory/questions/attribute/color_question.csv 

python data/ToolTrajectory/question_generation/generate_deerapi_with_json_color_question.py \
-prompt data/ToolTrajectory/prompts/attribute/prompts_comparative_color_qa_2.txt \
-type color \
-root data/HM3D/ \
-output $TMP_FILE

python data/ToolTrajectory/question_generation/generate_deerapi_with_images_color_answer.py \
-prompt data/ToolTrajectory/prompts/attribute/prompts_comparative_color_onlyans.txt \
-root data/HM3D \
-input $TMP_FILE \
-output data/ToolTrajectory/questions/attribute/color.csv
