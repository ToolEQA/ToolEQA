from transformers import Tool
import os
import json
from PIL import Image

class ObjectCrop(Tool):
    name = "ObjectCrop"
    description = "Given the bounding boxes of objects, crop and save the relevant objects from the image."
    inputs = {
        "bound_boxes": {"description": "the bounding boxes of objects", "type": "string"},
        "image_path": {
            "description": "The path to the image on which to crop objects.",
            "type": "string",
        },
    }
    output_type = "string"


    def forward(self, bound_boxes: str, image_path: str) -> list:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

        # 解析 bounding boxes
        try:
            boxes = json.loads(bound_boxes)
        except Exception as e:
            raise ValueError(f"Bounding box string is not valid JSON: {e}")

        if not isinstance(boxes, list) or not all(len(box) == 4 for box in boxes):
            raise ValueError("Bounding boxes must be a list of [x1, y1, x2, y2]")

        # 裁剪图像并保存
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = f"{base_name}_crops"
        os.makedirs(output_dir, exist_ok=True)

        output_paths = []
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            cropped = image.crop((x1, y1, x2, y2))
            output_path = os.path.join(output_dir, f"{base_name}_obj_{idx}.jpg")
            cropped.save(output_path)
            output_paths.append(output_path)

        return output_paths