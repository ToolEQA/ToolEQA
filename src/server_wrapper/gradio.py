import json
from pathlib import Path
from typing import Tuple, List, Dict, Any

import gradio as gr
from PIL import Image

ROOT = Path(__file__).parent
IMAGE_ROOT = "cache/EQA-RT-Seen.zs"
RUN_JSON_PATH = "results/EQA-RT-Seen.zs/result_0.jsonl"


def load_run(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    step_dict = {s["step"]: s for s in data["steps"]}
    step_indices = sorted(step_dict.keys())
    return data, step_dict, step_indices


DATA, STEP_DICT, STEP_INDICES = load_run(RUN_JSON_PATH)


def get_step_info(step_idx: int):
    """给 Gradio 回调用：根据 step index 返回要展示的内容。"""
    step = STEP_DICT.get(step_idx)
    if step is None:
        return None, None, "No step", "No code", "No observations"

    # "./cache/EQA-RT-Seen.zs/bKOs7RyRRxOW8uXFgfHYAg/0.png"
    rgb_img = Image.open(step["image"]) if (step["image"]).exists() else None
    # "./cache/EQA-RT-Seen.zs/bKOs7RyRRxOW8uXFgfHYAg/map_0.png"
    map_path = step["image"].parent / f"map_{step_idx}.png"
    map_img = Image.open(map_path) if (map_path).exists() else None

    # Thoughts 文本
    thoughts = step.get("thoughts", [])
    if thoughts:
        thoughts_text = "\n".join([f"[Thought {i}] {t}" for i, t in enumerate(thoughts)])
    else:
        thoughts_text = "No thoughts."

    # Code 文本（这里先简单拼在一起）
    codes = step.get("code", [])
    if codes:
        code_texts = []
        for j, code in enumerate(codes):
            lang = code.get("language", "text")
            code_texts.append(f"# Code {j} ({lang})\n{code['content']}")
        code_text = "\n\n".join(code_texts)
    else:
        code_text = "No code."

    # Observations 文本
    obs = step.get("observations", [])
    if obs:
        obs_text = "\n".join([f"- {o}" for o in obs])
    else:
        obs_text = "No observations."

    # 你也可以把 extra 单独做一个文本框展示
    extra = step.get("extra")
    if extra:
        extra_text = json.dumps(extra, ensure_ascii=False, indent=2)
        obs_text = obs_text + "\n\n[Extra]\n" + extra_text

    return rgb_img, map_img, thoughts_text, code_text, obs_text


with gr.Blocks(title="Model Run Viewer") as demo:
    gr.Markdown(f"# Model Run Viewer: {DATA.get('run_id', 'N/A')}")

    with gr.Row():
        step_slider = gr.Slider(
            minimum=STEP_INDICES[0],
            maximum=STEP_INDICES[-1],
            value=STEP_INDICES[0],
            step=1,
            label="Step",
        )

    with gr.Row():
        rgb_out = gr.Image(label="RGB", type="pil")
        map_out = gr.Image(label="Map", type="pil")

    with gr.Row():
        thoughts_out = gr.Textbox(label="Thoughts", lines=8)
    with gr.Row():
        code_out = gr.Code(label="Code", language="python")  # 简单先指定 python
    with gr.Row():
        obs_out = gr.Textbox(label="Observations & Extra", lines=8)

    # 绑定交互
    step_slider.change(
        fn=get_step_info,
        inputs=step_slider,
        outputs=[rgb_out, map_out, thoughts_out, code_out, obs_out],
    )

    # 启动时先展示第一个 step
    demo.load(
        fn=get_step_info,
        inputs=step_slider,
        outputs=[rgb_out, map_out, thoughts_out, code_out, obs_out],
    )

if __name__ == "__main__":
    demo.launch()
