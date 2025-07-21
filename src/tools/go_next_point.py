from transformers import Tool
from src.runs.eqa_modeling import EQA_Modeling
from omegaconf import OmegaConf
import cv2

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
        self.debug = kwargs.get("debug", False)
        if self.debug:
            return
        
        cfg = OmegaConf.load("/home/zml/algorithm/ReactEQA/config/react-eqa.yaml")
        OmegaConf.resolve(cfg)
        self.eqa_modeling = EQA_Modeling(cfg)
        
        self.base_path = "./cache/next_point_{}.jpg"
        self.step_idx = -1

        self.cur_rgb_path = None

    def initialize(self, data):
        self.cur_rgb_path = self.eqa_modeling.initialize(data)
        self.step_idx = 0

    def forward(self, command):
        if self.debug:
            return "./cache/init_rgb.png"
        
        self.step_idx += 1
        save_path = self.base_path.format(self.step_idx)
        self.eqa_modeling.go_next_point(command)
        cv2.imwrite(save_path, cv2.cvtColor(self.eqa_modeling.cur_rgb, cv2.COLOR_RGB2BGR))
        self.cur_rgb_path = save_path
        return save_path