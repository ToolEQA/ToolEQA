
from src.tools.vqa import VQATool
from src.tools.detect_2d import Detect2DTool
from src.tools.detect_3d import Detect3DTool
from src.tools.go_next_point import GoNextPointTool
from src.tools.object_location import ObjectLocationTool
from src.tools.query_object_info import QueryObjectInfoTool
from src.tools.register_view import RigisterViewTool
from src.tools.segment_instance import SegTool
from src.tools.final_answer import FinalAnswerTool

from transformers.agents.tools import get_tool_description_with_args

def get_tool_box():
    MODEL_TOOLBOX = [
        VQATool(),
        Detect2DTool(),
        Detect3DTool(),
        GoNextPointTool(),
        ObjectLocationTool(),
        QueryObjectInfoTool(),
        RigisterViewTool(),
        SegTool(),
        FinalAnswerTool(),
    ]
    return MODEL_TOOLBOX

def show_tool_descriptions(tools):
    return "\n".join(
            [get_tool_description_with_args(tool) for tool in tools]
        )

if __name__=="__main__":
    tool_box = get_tool_box()
    tool_desc = show_tool_descriptions(tool_box)
    print(tool_desc)
    