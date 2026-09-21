from src.tools.compat import Tool
from src.llm_engine.qwen import QwenEngine
from src.llm_engine.gpt import GPTEngine
import os
from omegaconf import OmegaConf

class VisualQATool(Tool):
    name = "VisualQATool"
    description = (
        "Answer a question about one or more observed images. Use exact paths returned by "
        "GoNextPointTool or ObjectCrop."
    )
    inputs = {
        "question": {"description": "the question to answer", "type": "string"},
        "image_paths": {
            "description": "The path to the image on which to answer the question",
            "type": "string",
        },
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = kwargs.get("debug", False)
        self.gpu_id = kwargs.get("gpu_id", 0)
        self.args = kwargs.get("args", None)
        self.cfg = None
        self.client = None
        self.last_resolved_image_paths = []
        self.navigation_tool = None
        if self.debug:
            return

    def _ensure_client(self):
        if self.debug or self.client is not None:
            return
        self.cfg = OmegaConf.load(self.args.cfg)
        OmegaConf.resolve(self.cfg)
        self.client = QwenEngine("/mynvme0/models/Qwen/Qwen2.5-VL-3B-Instruct", device=f"cuda:{self.gpu_id}")
        # self.client = GPTEngine("gpt-4o-mini")

    def bind_navigation_tool(self, navigation_tool):
        self.navigation_tool = navigation_tool

    def _resolve_image_path(self, image_path: str) -> str:
        if self.navigation_tool is not None:
            return self.navigation_tool.resolve_image_path(image_path, fallback_to_current=True)
        candidates = [image_path]
        if not os.path.isabs(image_path):
            candidates.extend([os.path.join(self.cfg.output_dir, image_path), os.path.join(os.sep, image_path)])
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.realpath(os.path.abspath(candidate))
        raise FileNotFoundError(f"No usable image is available for {image_path!r}")

    def forward_qwen(self, question, image_paths) -> str:
        self._ensure_client()
        add_note = False
        if type(question) is not str:
            raise Exception("parameter question should be a string.")
        if not question:
            add_note = True
            question = "Please write a detailed caption for this image."

        if isinstance(image_paths, str):
            image_paths = [image_paths]
        elif isinstance(image_paths, list):
            image_paths = image_paths
        else:
            print ('The type of input image is ', type(image_paths))
            raise Exception("The type of input image should be string (image path)")

        resolved_image_paths = []
        for image_path in image_paths:
            resolved_image_paths.append(self._resolve_image_path(image_path))
        self.last_resolved_image_paths = resolved_image_paths

        messages = [
            {"role": "user", "content": question}
        ]
        output = self.client.call_vlm(
            messages,
            image_paths = resolved_image_paths
        )

        if add_note:
            output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

        return output

    def forward(self, question, image_path="", image_paths=""):
        if self.debug:
            return "This is a debug context."
        self._ensure_client()
        if image_path != "":
            return self.forward_qwen(question, image_path)
        elif image_paths != "":
            return self.forward_qwen(question, image_paths)
