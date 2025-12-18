import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import gradio as gr
from PIL import Image

ROOT = Path(__file__).parent
EXP_NAME = "EQA-RT-Seen.zs"  # 这里是实验名称，通常是数据集名称
JSONL_ROOT = f"results/{EXP_NAME}"   # 多条数据，每行一个 JSON


def load_all_traces(jsonl_root: Path) -> List[Dict[str, Any]]:
    traces = []
    jsonl_root = Path(jsonl_root)
    for jsonl_path in jsonl_root.iterdir():
        if jsonl_path != jsonl_root / "eval_results.jsonl":
            continue
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                trace = json.loads(line)
                react = trace["summary"]["react"]
                if len(react) == 0:
                    continue
                if "final_answer" in react[-1]["code"]:
                    traces.append(trace)
    return traces


def split_react_into_steps(react: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按 GoNextPointTool 切成逻辑 step。
    返回:
    [
      {
        "step_index": 0,
        "entries": [ {thought, code, observation}, ... ],
        "go_call_index": 0  # 第几次调用 GoNextPointTool（从0开始）
      },
      ...
    ]
    """
    steps: List[Dict[str, Any]] = []
    current_entries: List[Dict[str, Any]] = []
    go_call_count = 0  # 已出现的 GoNextPointTool 次数

    def flush_step():
        nonlocal current_entries, go_call_count
        if not current_entries:
            return
        step_index = len(steps)
        go_index = go_call_count - 1 if go_call_count > 0 else 0
        steps.append(
            {
                "step_index": step_index,
                "entries": list(current_entries),
                "go_call_index": max(go_index, 0),
            }
        )
        current_entries.clear()

    for entry in react:
        current_entries.append(entry)
        code = (entry.get("code") or "")
        if "GoNextPointTool" in code or "final_answer" in code:
            go_call_count += 1
            flush_step()

    # 末尾如果没有再遇到 GoNextPointTool，也单独作为一个 step
    if current_entries:
        go_index = go_call_count - 1 if go_call_count > 0 else 0
        steps.append(
            {
                "step_index": len(steps),
                "entries": list(current_entries),
                "go_call_index": max(go_index, 0),
            }
        )

    return steps


def format_entries(entries: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    """
    把一个逻辑 step 中的多条 react entry 整理成三块文本：
    - thoughts_text
    - code_text
    - obs_text
    """
    thoughts_lines = []
    code_blocks = []
    obs_lines = []

    for i, e in enumerate(entries):
        th = (e.get("thought") or "").strip()
        cd = (e.get("code") or "").strip()
        ob = (e.get("observation") or "").strip()

        if th:
            thoughts_lines.append(f"[{i}] {th}")
        if cd:
            code_blocks.append(f"# Block {i}\n{cd}")
        if ob:
            obs_lines.append(f"[{i}] {ob}")

    thoughts_text = "\n\n".join(thoughts_lines) if thoughts_lines else "No thoughts."
    code_text = "\n\n\n".join(code_blocks) if code_blocks else "No code."
    obs_text = "\n\n".join(obs_lines) if obs_lines else "No observations."

    return thoughts_text, code_text, obs_text


# 预加载所有样本
ALL_TRACES = load_all_traces(JSONL_ROOT)
NUM_SAMPLES = len(ALL_TRACES)


def prepare_sample(idx: int) -> Dict[str, Any]:
    """根据样本 index 准备该样本的数据（react steps + step 数组）"""
    trace = ALL_TRACES[idx]
    meta = trace.get("meta", {})
    sample_id = meta.get("sample_id", trace.get("sample_id", "unknown"))
    react = trace["summary"]["react"]
    logic_steps = split_react_into_steps(react)

    raw_steps = trace.get("step", [])  # 原始的 step 数组，里面有 init + 后续 step
    image_root = Path(raw_steps[0]["image"]).parent if raw_steps else Path("")
    init_step = [{
        "step": -1,
        "image": f"cache/{EXP_NAME}/{sample_id}/init.png"
    }]
    raw_steps = init_step + raw_steps

    return {
        "trace": trace,
        "logic_steps": logic_steps,
        "raw_steps": raw_steps,
    }


def get_step_images(
    raw_steps: List[Dict[str, Any]],
    go_call_index: int,
) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    """
    返回 (rgb_img, map_img, info_text)

    约定：
    - raw_steps[0]["image"] 是 init.png（只有 RGB，没有 map）
    - 对于第 k 次 GoNextPointTool（k 从 1 开始计数），对应 raw_steps[k]["image"] 和 raw_steps[k]["map_image"]
      所以这里传入的 go_call_index = 第几次 GoNextPointTool（从 0 开始）：
        - go_call_index == 0 表示逻辑 step 0：显示 init.png（raw_steps[0]），map 为空
        - go_call_index >= 1：显示 raw_steps[go_call_index]["image"] 和 ["map_image"]
    """
    if not raw_steps:
        return None, None, "No step info in this sample."

    # init 索引
    init_idx = 0
    
    # 逻辑 step 0 => init.png
    if go_call_index == 0:
        if init_idx >= len(raw_steps):
            return None, None, "No init step in raw_steps."
        step_obj = raw_steps[init_idx]
        rgb_path_str = step_obj.get("image")
        map_path_str = None  # init 没有 map
    else:
        # 第 go_call_index 次 GoNextPointTool 使用 raw_steps[go_call_index]
        if go_call_index >= len(raw_steps):
            return None, None, f"No step for GoNextPointTool index {go_call_index}."
        step_obj = raw_steps[go_call_index]
        rgb_path_str = step_obj.get("image")
        
        rgb_path = Path(rgb_path_str)
        # 如果你的 map 字段名不是 "map_image"，在这里改，例如 step_obj.get("map") / step_obj.get("heatmap")
        map_path_str = rgb_path.parent / f"map_{rgb_path.stem.split('_')[-1]}.png"

    rgb_img = None
    map_img = None
    msg_parts = []

    def resolve_path(p_str: Optional[str]) -> Optional[Path]:
        if not p_str:
            return None
        p = Path(p_str)
        # if not p.is_absolute():
        #     p = (ROOT / p).resolve()
        return p

    # RGB 图
    if rgb_path_str:
        p_rgb = resolve_path(rgb_path_str)
        if p_rgb and p_rgb.exists():
            try:
                rgb_img = Image.open(p_rgb)
                msg_parts.append(f"RGB: {p_rgb}")
            except Exception as e:
                msg_parts.append(f"RGB fail {p_rgb}: {e}")
        else:
            msg_parts.append(f"RGB not found: {p_rgb}")
    else:
        msg_parts.append("RGB path missing")

    # MAP 图（init 时是 None）
    if map_path_str:
        p_map = resolve_path(map_path_str)
        if p_map and p_map.exists():
            try:
                map_img = Image.open(p_map)
                msg_parts.append(f"MAP: {p_map}")
            except Exception as e:
                msg_parts.append(f"MAP fail {p_map}: {e}")
        else:
            msg_parts.append(f"MAP not found: {p_map}")
    else:
        msg_parts.append("MAP path missing or init step (no map).")

    info_text = " | ".join(msg_parts)
    return rgb_img, map_img, info_text


def get_step_view(sample_idx: int, step_idx: int):
    """
    核心渲染：给定样本 & 希望查看的 step index，
    返回给前端显示的内容 + 实际使用的 step_idx（可能被 clamp）。
    """

    if NUM_SAMPLES == 0:
        return (
            "No data.",          # header_md
            "",                  # sample_info_md
            None,                # rgb image
            None,                # map image
            "No image.",         # image_info_md
            "No thoughts.",      # thoughts_text
            "No code.",          # code_text
            "No observations.",  # obs_text
            0,                   # corrected step idx
        )

    letters = ["A", "B", "C", "D"]

    # 样本 index 限制在合法范围
    sample_idx = max(0, min(sample_idx, NUM_SAMPLES))
    sample_data = prepare_sample(sample_idx)
    trace = sample_data["trace"]
    logic_steps = sample_data["logic_steps"]
    raw_steps = sample_data["raw_steps"]

    if not logic_steps:
        header_md = f"### Sample {sample_idx + 1}/{NUM_SAMPLES + 1}"
        meta = trace.get("meta", {})
        question = meta.get("question", "") or trace.get("question", "")

        answer_choice = meta.get('answer', "")
        proposals = meta.get('proposals', [])
        answer_str = proposals[letters.index(answer_choice)]
        sample_info_md = (
            f"**Sample ID:** `{meta.get('sample_id', 'N/A')}`  \n"
            f"**Question:** {question} \n"
            f"**Final Answer:** {trace.get('summary', {}).get('final_answer', '')}\n\n"
            f"**GT Answer:** {answer_str}\n\n"
            f"⚠️ This sample has no logic steps (no GoNextPointTool found)."
        )
        # 这个样本没有 step，就强制 step_idx = 0
        return (
            header_md,
            sample_info_md,
            None,
            None,
            "No image for this sample.",
            "No thoughts.",
            "No code.",
            "No observations.",
            0,
        )

    # step index 限制在合法范围（关键：这里会 clamp）
    step_idx = max(0, min(step_idx, len(logic_steps) - 1))
    s = logic_steps[step_idx]

    # 顶部信息
    meta = trace.get("meta", {})
    question = meta.get("question", "") or trace.get("question", "")
    final_answer = trace.get("summary", {}).get("final_answer", "")

    answer_choice = meta.get('answer', "")
    proposals = meta.get('proposals', [])
    answer_str = proposals[letters.index(answer_choice)]

    header_md = f"### Sample {sample_idx + 1}/{NUM_SAMPLES}  |  Step {step_idx + 1}/{len(logic_steps)}"
    sample_info_md = (
        f"**Sample ID:** `{meta.get('sample_id', 'N/A')}`  \n"
        f"**Question:** {question}\n\n"
        f"**Final Answer:** {final_answer}\n\n"
        f"**GT Answer:** {answer_str}\n"
    )

    # 根据 go_call_index 去 step[...] 里拿两张图
    go_call_index = s.get("go_call_index", 0)
    rgb_img, map_img, img_info = get_step_images(raw_steps, go_call_index)

    # thoughts, code, observations
    thoughts_text, code_text, obs_text = format_entries(s["entries"])

    return (
        header_md,
        sample_info_md,
        rgb_img,
        map_img,
        img_info,
        thoughts_text,
        code_text,
        obs_text,
        step_idx,   # 把 clamp 后的 index 返回，用来回写 slider
    )


# ============ 搭 Gradio 界面 ============

with gr.Blocks(title="JSONL React Viewer (RGB + Map)") as demo:
    gr.Markdown("# JSONL React Navigation Viewer (RGB + Map)")

    current_sample_idx = gr.State(0)
    current_step_idx = gr.State(0)

    header_md = gr.Markdown("")
    sample_info_md = gr.Markdown("")

    with gr.Row():
        prev_btn = gr.Button("⬅️ Prev Sample")
        next_btn = gr.Button("Next Sample ➡️")

    # 初始先给个固定范围，之后用 gr.update 覆盖
    with gr.Row():
        step_slider = gr.Slider(
            minimum=0,
            maximum=49,
            value=0,
            step=1,
            label="Step (per-sample)",
            interactive=True,
        )

    # 两张图并排：左 RGB 右 map
    with gr.Row():
        rgb_img_out = gr.Image(label="RGB Image", type="pil")
        map_img_out = gr.Image(label="Map Image", type="pil")

    with gr.Row():
        img_info_md = gr.Markdown()

    with gr.Row():
        thoughts_out = gr.Textbox(
            label="Thoughts in this step",
            lines=10,
        )

    with gr.Row():
        code_out = gr.Code(
            label="Code blocks in this step",
            language="python",
        )

    with gr.Row():
        obs_out = gr.Textbox(
            label="Observations in this step",
            lines=10,
        )

    # ---- 回调逻辑 ----

    def on_change_step(step_idx, sample_idx):
        # slider 变化时，重新渲染这个样本的该 step
        (
            h,
            info,
            rgb_img,
            map_img,
            img_info,
            th,
            cd,
            obs,
            corrected_idx,
        ) = get_step_view(int(sample_idx), int(step_idx))

        # 计算当前样本的最大 step index
        sample_data = prepare_sample(int(sample_idx))
        logic_steps = sample_data["logic_steps"]
        max_step = max(0, len(logic_steps) - 1)

        return (
            h,
            info,
            rgb_img,
            map_img,
            img_info,
            th,
            cd,
            obs,
            gr.update(   # 更新 slider 的值和最大值
                value=corrected_idx,
                minimum=0,
                maximum=max_step,
            ),
        )

    def on_prev_sample(sample_idx, step_idx):
        new_sample_idx = int(sample_idx) - 1
        if new_sample_idx < 0:
            new_sample_idx = 0
        # 切样本时，默认展示这个样本的 step 0
        (
            h,
            info,
            rgb_img,
            map_img,
            img_info,
            th,
            cd,
            obs,
            corrected_idx,
        ) = get_step_view(new_sample_idx, 0)

        sample_data = prepare_sample(new_sample_idx)
        logic_steps = sample_data["logic_steps"]
        max_step = max(0, len(logic_steps) - 1)

        return (
            new_sample_idx,    # 更新 current_sample_idx
            corrected_idx,     # 更新 current_step_idx
            h,
            info,
            rgb_img,
            map_img,
            img_info,
            th,
            cd,
            obs,
            gr.update(        # 更新 slider 的值和最大值
                value=corrected_idx,
                minimum=0,
                maximum=max_step,
            ),
        )

    def on_next_sample(sample_idx, step_idx):
        new_sample_idx = int(sample_idx) + 1
        if new_sample_idx >= NUM_SAMPLES:
            new_sample_idx = NUM_SAMPLES - 1
        (
            h,
            info,
            rgb_img,
            map_img,
            img_info,
            th,
            cd,
            obs,
            corrected_idx,
        ) = get_step_view(new_sample_idx, 0)

        sample_data = prepare_sample(new_sample_idx)
        logic_steps = sample_data["logic_steps"]
        max_step = max(0, len(logic_steps) - 1)

        return (
            new_sample_idx,
            corrected_idx,
            h,
            info,
            rgb_img,
            map_img,
            img_info,
            th,
            cd,
            obs,
            gr.update(
                value=corrected_idx,
                minimum=0,
                maximum=max_step,
            ),
        )

    # slider 改变：更新内容 + 回写纠正后的 step_idx
    step_slider.change(
        fn=on_change_step,
        inputs=[step_slider, current_sample_idx],
        outputs=[
            header_md,
            sample_info_md,
            rgb_img_out,
            map_img_out,
            img_info_md,
            thoughts_out,
            code_out,
            obs_out,
            step_slider,        # 用函数返回的 gr.update 更新 slider
        ],
    ).then(
        fn=lambda v: int(v),
        inputs=step_slider,
        outputs=current_step_idx,
    )

    # 上一个样本
    prev_btn.click(
        fn=on_prev_sample,
        inputs=[current_sample_idx, current_step_idx],
        outputs=[
            current_sample_idx,
            current_step_idx,
            header_md,
            sample_info_md,
            rgb_img_out,
            map_img_out,
            img_info_md,
            thoughts_out,
            code_out,
            obs_out,
            step_slider,    # 设置 slider 的显示值和最大值
        ],
    )

    # 下一个样本
    next_btn.click(
        fn=on_next_sample,
        inputs=[current_sample_idx, current_step_idx],
        outputs=[
            current_sample_idx,
            current_step_idx,
            header_md,
            sample_info_md,
            rgb_img_out,
            map_img_out,
            img_info_md,
            thoughts_out,
            code_out,
            obs_out,
            step_slider,
        ],
    )

    # 初始加载：样本 0，step 0
    def on_app_load():
        (
            h,
            info,
            rgb_img,
            map_img,
            img_info,
            th,
            cd,
            obs,
            corrected_idx,
        ) = get_step_view(0, 0)

        sample_data = prepare_sample(0)
        logic_steps = sample_data["logic_steps"]
        max_step = max(0, len(logic_steps) - 1)

        return (
            0,               # current_sample_idx
            corrected_idx,   # current_step_idx
            h,
            info,
            rgb_img,
            map_img,
            img_info,
            th,
            cd,
            obs,
            gr.update(       # 初始化 slider 的值和最大值
                value=corrected_idx,
                minimum=0,
                maximum=max_step,
            ),
        )

    demo.load(
        fn=on_app_load,
        inputs=None,
        outputs=[
            current_sample_idx,
            current_step_idx,
            header_md,
            sample_info_md,
            rgb_img_out,
            map_img_out,
            img_info_md,
            thoughts_out,
            code_out,
            obs_out,
            step_slider,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # 监听所有网卡
        server_port=7860,        # 端口号，自定一个没被占用的即可
        share=False,             # 不用 Gradio 的公网分享
    )
