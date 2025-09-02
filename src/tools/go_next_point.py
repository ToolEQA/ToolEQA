from transformers import Tool
from src.runs.eqa_modeling import EQA_Modeling
from omegaconf import OmegaConf
import cv2
import os

class GoNextPointTool(Tool):
    name = "GoNextPointTool"
    description = "the agent conitnue explore next point and obtain next observation (rgb image)."
    inputs = {
        "query": {
            "description": "Next exploration direction, ONLY [move_forward, turn_left, turn_right] are supported.",
            "type": "string",
        },
    }
    output_type = "image"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_id = kwargs.get("gpu_id", 0)
        self.debug = kwargs.get("debug", False)
        self.args = kwargs.get("args", None)
        if self.debug:
            return
        
        self.cfg = OmegaConf.load(self.args.cfg)
        OmegaConf.resolve(self.cfg)
        self.eqa_modeling = EQA_Modeling(self.cfg, self.gpu_id)

        self.step_idx = -1

        self.cur_rgb_path = None

    def initialize(self, data):
        self.cur_rgb_path = self.eqa_modeling.initialize(data)
        self.sample_id = data['sample_id']
        self.save_dir = os.path.join(self.cfg.output_dir, self.sample_id)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.step_idx = 0

    def forward(self, command):
        if self.debug:
            return "./cache/init_rgb.png"
        
        self.step_idx += 1
        save_path = os.path.join(self.save_dir, f"next_point_{self.step_idx}.jpg")
        self.eqa_modeling.go_next_point(command)
        cv2.imwrite(save_path, cv2.cvtColor(self.eqa_modeling.cur_rgb, cv2.COLOR_RGB2BGR))
        self.cur_rgb_path = os.path.abspath(save_path)
        return self.cur_rgb_path