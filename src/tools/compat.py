try:
    from transformers import Tool as HFTool
except Exception:
    HFTool = None


class _FallbackTool:
    name = ""
    description = ""
    inputs = {}
    output_type = "string"

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


Tool = HFTool or _FallbackTool
