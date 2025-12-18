from transformers import Tool
import os
import json
from PIL import Image

class ObjectCrop(Tool):
    name = "ObjectCrop"
    description = "Given the bounding boxes of objects, crop and save the relevant objects from the image."
    inputs = {
        "bound_boxes": {"description": "the bounding boxes of objects", "type": "number"},
        "image_path": {
            "description": "The path to the image on which to crop objects.",
            "type": "string",
        },
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_id = kwargs.get("gpu_id", 0)
        self.debug = kwargs.get("debug", False)
        self.image_root = "data/EQA-Traj-0720"
        if self.debug:
            return

    def list_dim(self, lst):
        if not isinstance(lst, list):
            return 0
        if not lst:
            return 1  # 空列表，当作一维
        return 1 + self.list_dim(lst[0])

    def forward(self, bounding_box: list, image_path: str) -> list:
        if self.debug:
            return "./cache/init_crop.png"
        
        try:
            # image_path = os.path.join(self.image_root, image_path)
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

        d_list = self.list_dim(bounding_box)
        if d_list == 1:
            bounding_box = [bounding_box]
            
        if not isinstance(bounding_box, list) or not all(len(box) == 4 for box in bounding_box):
            raise ValueError("Bounding boxes must be a list of [x1, y1, x2, y2]")

        # 裁剪图像并保存
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        folder = os.path.dirname(image_path)
        # output_dir = f"{base_name}_crops"
        # os.makedirs(output_dir, exist_ok=True)

        output_paths = []
        for idx, bbox in enumerate(bounding_box):
            cropped = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            output_path = os.path.join(folder, f"{base_name}_crop_obj_{idx}.jpg")
            cropped.save(output_path)
            output_paths.append(output_path)

        return output_paths