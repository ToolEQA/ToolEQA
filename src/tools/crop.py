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
        self.debug = kwargs.get("debug", False)
        if self.debug:
            return

    def forward(self, bounding_box: list, image_path: str) -> list:
        if self.debug:
            return "./cache/init_crop.png"
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

        # # 解析 bounding boxes
        # try:
        #     boxes = json.loads(boubounding_boxnd_box)
        # except Exception as e:
        #     raise ValueError(f"Bounding box string is not valid JSON: {e}")

        if not isinstance(bounding_box, list) or not all(len(box) == 4 for box in bounding_box):
            raise ValueError("Bounding boxes must be a list of [x1, y1, x2, y2]")

        # 裁剪图像并保存
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # output_dir = f"{base_name}_crops"
        # os.makedirs(output_dir, exist_ok=True)

        output_paths = []
        for idx, bbox in enumerate(bounding_box):
            cropped = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            output_path = f"./cache/{base_name}_obj_{idx}.jpg"
            cropped.save(output_path)
            output_paths.append(output_path)

        return output_paths