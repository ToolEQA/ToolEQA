from src.tools.vqa import VisualQATool
from src.tools.location_2d import ObjectLocation2D
from src.tools.location_3d import ObjectLocation3D
from src.tools.go_next_point import GoNextPointTool
from src.tools.segment_instance import SegmentInstanceTool
from src.tools.final_answer import FinalAnswerTool
from src.tools.crop import ObjectCrop

from transformers.agents.tools import get_tool_description_with_args

def get_tool_box(debug=False, tool_box_selected = None):
    if debug:
        if tool_box_selected is None:
            MODEL_TOOLBOX = [
                VisualQATool(debug=debug),
                ObjectLocation2D(debug=debug),
                ObjectLocation3D(debug=debug),
                GoNextPointTool(debug=debug),
                SegmentInstanceTool(debug=debug),
                FinalAnswerTool(debug=debug),
                ObjectCrop(debug=debug)
            ]
        else:
            MODEL_TOOLBOX = []
            for tb in tool_box_selected:
                MODEL_TOOLBOX.append(type(tb)(debug=debug))
    else:
        if tool_box_selected is None:
            MODEL_TOOLBOX = [
                VisualQATool(),
                ObjectLocation2D(),
                ObjectLocation3D(),
                GoNextPointTool(),
                SegmentInstanceTool(),
                FinalAnswerTool(),
                ObjectCrop()
            ]
        else:
            MODEL_TOOLBOX = []
            for tb in tool_box_selected:
                MODEL_TOOLBOX.append(tb)
    return MODEL_TOOLBOX


# from src.tools.vqa import VQATool
# from src.tools.detect_2d import Detect2DTool
# from src.tools.detect_3d import Detect3DTool
# from src.tools.go_next_point import GoNextPointTool
# from src.tools.object_location import ObjectLocationTool
# from src.tools.segment_instance import SegTool
# from src.tools.final_answer import FinalAnswerTool

# from transformers.agents.tools import get_tool_description_with_args

# def get_tool_box():
#     MODEL_TOOLBOX = [
#         VQATool(),
#         Detect2DTool(),
#         Detect3DTool(),
#         GoNextPointTool(),
#         ObjectLocationTool(),
#         SegTool(),
#         FinalAnswerTool(),
#     ]
#     return MODEL_TOOLBOX

def show_tool_descriptions(tools):
    return "\n".join(
            [get_tool_description_with_args(tool) for tool in tools]
        )

if __name__=="__main__":
    tool_box = get_tool_box(debug=True)
    tool_desc = show_tool_descriptions(tool_box)
    print(tool_desc)
