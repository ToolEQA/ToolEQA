from transformers import Tool
from src.runs.eqa_modeling import EQA_Modeling
from omegaconf import OmegaConf
import cv2

class GoNextPointTool(Tool):
    name = "GoNextPointTool"
    description = "the agent conitnue explore next point and obtain next observation (rgb image)."
    inputs = {
        "query": {
            "description": "Next exploration direction",
            "type": "string",
        },
    }
    output_type = "image"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cfg = OmegaConf.load("config/react-eqa.yaml")
        OmegaConf.resolve(cfg)
        self.eqa_modeling = EQA_Modeling(cfg)
        
        self.base_path = "./tmp/next_point_{}.jpg"
        self.step_idx = -1

    def set_data(self, data):
        self.eqa_modeling.initialize(data)

    def forward(self):
        self.step_idx += 1
        save_path = self.base_path.format(self.step_idx)
        self.eqa_modeling.go_next_point()
        cv2.imwrite(save_path, self.eqa_modeling.cur_rgb)
        return save_path