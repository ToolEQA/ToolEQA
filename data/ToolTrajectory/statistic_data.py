# 统计每个问题有多少步探索
# 统计每个问题涉及多少obj

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def draw_multi_linechart(data_dicts, labels, caption, x_label, y_label, save_path):
    """
    data_dicts: [dict1, dict2, ...]   每个 dict 的 key 为 x 轴，value 为数量
    labels:    [label1, label2, ...]  每组数据的名字
    """
    # 所有 key 的全集（保证顺序一致）
    all_keys = sorted(set().union(*[d.keys() for d in data_dicts]))
    x = np.arange(len(all_keys))
    values = [diff[k] for k in all_keys]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in values]
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.7)

    # X 轴放在 y=0
    step = max(1, len(all_keys) // 8)   # 控制最多显示 15 个刻度
    ax.set_xticks(x[::step] + 0.35 * (len(data_dicts)-1) / 2)
    ax.set_xticklabels(all_keys[::step], fontsize=14)

    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)

    # 添加数值标签
    # for bar in bars:
    #     height = bar.get_height()
    #     if height != 0:
    #         ax.text(bar.get_x() + bar.get_width()/2,
    #                 height + (0.5 if height > 0 else -0.5),
    #                 f"{height}",
    #                 ha="center",
    #                 va="bottom" if height > 0 else "top",
    #                 fontsize=10)

    # 美化
    ax.set_title(f"{caption} ({labels[0]} - {labels[1]})", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    # ax.set_xticks(x)
    # ax.set_xticklabels(all_keys, fontsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', transparent=True)

def draw_multi_histogram(data_dicts, labels, caption, x_label, y_label, save_path, show_bar=False):
    """
    data_dicts: [dict1, dict2, ...]   每个dict的key为x轴，value为数量
    labels:    [label1, label2, ...]  每组数据的名字
    """

    # 所有key的全集
    all_keys = sorted(set().union(*[d.keys() for d in data_dicts]))
    x = np.arange(len(all_keys))  # x轴位置

    # 创建画布
    fig, ax = plt.subplots(figsize=(7, 5))

    bar_width = 0.35  # 每组柱子的宽度
    colors = plt.cm.viridis(np.linspace(0.4, 0.9, len(data_dicts)))

    # 遍历每个dict画图
    for i, d in enumerate(data_dicts):
        values = [d.get(k, 0) for k in all_keys]
        bars = ax.bar(x + i * bar_width, values, width=bar_width,
                      color=colors[i], edgecolor="black", linewidth=0.7,
                      label=labels[i])

        if show_bar:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.2,
                            f"{height}", ha="center", va="bottom", fontsize=12)

    # 美化
    ax.set_title(caption, fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_xticks(x + bar_width * (len(data_dicts)-1) / 2)
    if not show_bar:
        step = max(1, len(all_keys) // 5)
        ax.set_xticks(x[::step] + bar_width * (len(data_dicts)-1) / 2)
        ax.set_xticklabels(all_keys[::step], fontsize=14)
    else:
        ax.set_xticklabels(all_keys, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', transparent=True)

def process_data(data):
    object_stats = {}
    thought_stats = {}
    data_count = len(data)
    for item in data:

        trajectory = item["trajectory"]
        related_objects = item["related_objects"]

        if len(related_objects) not in object_stats.keys():
            object_stats[len(related_objects)] = 0
        object_stats[len(related_objects)] += 1

        if len(trajectory) not in thought_stats.keys():
            thought_stats[len(trajectory)] = 0
        thought_stats[len(trajectory)] += 1

    object_stats = {k: object_stats[k] / data_count for k in sorted(object_stats.keys())}
    thought_count = {k: thought_stats[k] / data_count for k in sorted(thought_stats.keys())}

    # thought_stats_new = {}
    # thought_count = 0
    # idx = 0
    # for k in sorted(thought_stats.keys()):
    #     thought_count += thought_stats[k]
    #     if k % 5 == 0:
    #         thought_stats_new[idx] = thought_count / data_count
    #         thought_count = 0
    #         idx += 1
    
    # thought_stats_newnew = {}
    # thought_stats_newnew[0] = (thought_stats_new[0] + thought_stats_new[1]) / 2
    # thought_stats_newnew[1] = thought_stats_new[2]
    # thought_stats_newnew[2] = thought_stats_new[3]
    # thought_stats_newnew[3] = thought_stats_new[4]
    # thought_stats_newnew[4] = thought_stats_new[5]
    # thought_stats_newnew[5] = thought_stats_new[6] if 6 in thought_stats_new.keys() else 0

    # norm = 0
    # for k, v in thought_stats.items():
    #     norm += v
    # print(norm)

    return object_stats, thought_count


data_path = [
    "tmp/data/new_data/seen_testset.json",
    # "tmp/data/new_data/trainval.json",
    # "tmp/data/new_data/unseen_testset.json",
    ]

trainval_path = "tmp/data/new_data/trainval.json"
seen_path = "tmp/data/new_data/seen_testset.json"
unseen_path = "tmp/data/new_data/unseen_testset.json"

trainval_data = json.load(open(trainval_path, "r"))
seen_data = json.load(open(seen_path, "r"))
unseen_data = json.load(open(unseen_path, "r"))

# trainval_object_stats, trainval_thought_stats = process_data(trainval_data)
seen_object_stats, seen_thought_stats = process_data(seen_data)
unseen_object_stats, unseen_thought_stats = process_data(unseen_data)


# draw_multi_histogram([trainval_object_stats], ["train"],
#                      "question-object", "object count", "question count", "train_stats_objs.pdf", True)
# draw_multi_histogram([trainval_thought_stats], ["train"],
#                      "question-step", "step count", "question count", "train_stats_steps.pdf", False)

# draw_multi_histogram([seen_object_stats, unseen_object_stats], ["Seen", "Unseen"],
#                      "question-object", "object count", "question count", "stats_objs.pdf", True)
# draw_multi_histogram([seen_thought_stats, unseen_thought_stats], ["Seen", "Unseen"],
#                      "question-step", "step count", "question count", "stats_steps.pdf", False)

all_keys = sorted(set(seen_thought_stats.keys()) | set(unseen_thought_stats.keys()))
diff = {k: seen_thought_stats.get(k, 0) - unseen_thought_stats.get(k, 0) for k in all_keys}
draw_multi_linechart([diff], ["Seen", "Unseen"],
                     "Thought Step Proportion Difference", "", "Proportion Difference", "stats_steps.pdf")

# all_keys = sorted(set(seen_object_stats.keys()) | set(unseen_object_stats.keys()))
# diff = {k: seen_object_stats.get(k, 0) - unseen_object_stats.get(k, 0) for k in all_keys}
# draw_multi_linechart([diff], ["Seen", "Unseen"],
#                      "Object Proportion Difference", "", "Proportion Difference", "stats_objects.pdf")