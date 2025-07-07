from transformers import Tool
from src.runs.eqa_modeling import EQA_Modeling

class GoNextPointTool(Tool):
    name = "go_next_point"
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
        self.images = ["./data/EQA-Traj-0611/a6VtzZdES3Kxj42jjq58cg/0-0.png",
                "./data/EQA-Traj-0611/a6VtzZdES3Kxj42jjq58cg/0-1.png",
                "./data/EQA-Traj-0611/a6VtzZdES3Kxj42jjq58cg/0-2.png",
                "./data/EQA-Traj-0611/a6VtzZdES3Kxj42jjq58cg/0-3.png",
                "./data/EQA-Traj-0611/a6VtzZdES3Kxj42jjq58cg/0-4.png",
                "./data/EQA-Traj-0611/a6VtzZdES3Kxj42jjq58cg/0-5.png",
                "./data/EQA-Traj-0611/a6VtzZdES3Kxj42jjq58cg/0-6.png",
                "./data/EQA-Traj-0611/a6VtzZdES3Kxj42jjq58cg/1-1.png",
                "./data/EQA-Traj-0611/a6VtzZdES3Kxj42jjq58cg/1-2.png",
                "./data/EQA-Traj-0611/a6VtzZdES3Kxj42jjq58cg/1-3.png",]
        self.idx = -1

    def forward(self):
        self.idx += 1
        if self.idx < len(self.images):
            output = self.images[self.idx]
        else:
            output = self.images[-1]
        return output