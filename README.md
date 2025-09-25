# ToolEQA

# Installation
Set up the conda environment (Linux, Python 3.9):
```
conda env create -f environment.yml
conda activate explore-eqa
pip install -e .
```

Install the latest version of [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) (headless with no Bullet physics) with:
```
conda install habitat-sim headless -c conda-forge -c aihabitat
```

Install [flash-attention2](https://github.com/Dao-AILab/flash-attention):
```
pip install flash-attn --no-build-isolation
```

Install transformers for qwenvl
```
pip install git+https://github.com/huggingface/transformers
pip install qwen-vl-utils
```

Install AutoGPTQ
```
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install -vvv --no-build-isolation -e .
```

Install DetAny3D
```
git clone https://github.com/zmling22/DetAny3D.git
cd DetAny3D
# install SegmentAnything
pip install git+https://github.com/facebookresearch/segment-anything.git
# install UniDepth
git clone https://github.com/lpiccinelli-eth/UniDepth.git
cd UniDepth
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
cd ..
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
```

# Data Generation Pipeline
1. extract objects from scene
```
sh data/ToolTrajectory/gen_semantic/run_gen_sem_ddp.sh
```
2. generate and evaluate question and answer
```
# sample images
python data/ToolTrajectory/preprocessing/sampler.py
# generate question and answer
sh data/ToolTrajectory/question_generation/scripts/generate.sh
# evaluate
python data/ToolTrajectory/postprocessing/question_filter.py
```

3. generate trajectories
```
# save shorest path
python data/ToolTrajectory/postprocessing/trajectory.py
# generate thought and code
python data/ToolTrajectory/trajectory_gen/attribute/color/color_react_gen_step_withans.py
python data/ToolTrajectory/trajectory_gen/attribute/size/size_react_gen_step_withans.py
python data/ToolTrajectory/trajectory_gen/attribute/special/special_react_gen_step_withans.py
python data/ToolTrajectory/trajectory_gen/counting/counting_react_gen_step_withans.py
python data/ToolTrajectory/trajectory_gen/distance/distance_react_gen_step_withans.py
python data/ToolTrajectory/trajectory_gen/location/location/location_react_gen_step_withans_oneobject.py
python data/ToolTrajectory/trajectory_gen/location/location/location_react_gen_step_withans_twoobject.py
python data/ToolTrajectory/trajectory_gen/relationship/relationship_react_gen_step_withans.py
python data/ToolTrajectory/trajectory_gen/status/status_react_gen_step_withans.py
```

4. evaluate thought and code
```
sh data/ToolTrajectory/trajectory_gen/attribute/color/post_process.sh
sh data/ToolTrajectory/trajectory_gen/attribute/size/post_process.sh
sh data/ToolTrajectory/trajectory_gen/attribute/special/post_process.sh
sh data/ToolTrajectory/trajectory_gen/counting/post_process.sh
sh data/ToolTrajectory/trajectory_gen/distance/post_process.sh
sh data/ToolTrajectory/trajectory_gen/location/location/post_process_one.sh
sh data/ToolTrajectory/trajectory_gen/location/location/post_process_two.sh
sh data/ToolTrajectory/trajectory_gen/location/special/post_process.sh
sh data/ToolTrajectory/trajectory_gen/relationship/post_process.sh
sh data/ToolTrajectory/trajectory_gen/status/post_process.sh
```

# Train Qwen2.5VL
```
cd src/train/Qwen2.5-VL/qwen-vl-finetune/
sh scripts/sft_7b.sh
```

# Run ToolEQA
```
# run DetAny3D
cd DetAny3D
python app_mp.py
# run ToolEQA
cd ../ToolEQA
python src/agents/react_eqa_agent_mp.py --gpus 1,2,3,4
```

# Evaluation
```
python src/evaluation/eval_results_on_json.py
```