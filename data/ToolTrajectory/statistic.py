import pandas as pd
import os
import numpy as np
import json

from matplotlib import pyplot as plt
from collections import defaultdict
import matplotlib.colors as mcolors
import colorsys

def build_color_map(all_data_list):
    """
    构建颜色映射：
    - 外圈大类：均匀分布在色环上（高区分度）
    - 内圈小类：同色调的不同亮度
    """
    # 收集大类和小类
    outer_categories = set()
    inner_categories = {}
    for data in all_data_list:
        for outer in data.keys():
            if outer == "meta":
                continue
            outer_categories.add(outer)
            if outer not in inner_categories:
                inner_categories[outer] = set()
            inner_categories[outer].update(data[outer].keys())

    outer_categories = sorted(outer_categories)
    n_outer = len(outer_categories)

    color_map = {"outer": {}, "inner": {}}

    # 给大类分配颜色（均匀分布在 HSV 色环上）
    for i, outer in enumerate(outer_categories):
        hue = i / n_outer  # 均匀分布 [0,1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)  # 饱和度0.7, 明度0.9
        base_color = (r, g, b)
        color_map["outer"][outer] = base_color
        color_map["inner"][outer] = {}

        # 给子类分配同色不同亮度
        subcats = sorted(inner_categories[outer])
        m = len(subcats)
        for j, subcat in enumerate(subcats):
            factor = 0.5 + 0.5 * (j / max(1, m-1))  # 0.5 ~ 1.0
            rr = r*factor + (1-factor)
            gg = g*factor + (1-factor)
            bb = b*factor + (1-factor)
            lighter = (rr, gg, bb)
            color_map["inner"][outer][subcat] = lighter

    return color_map

def statistic(data_file):
    data = json.load(open(data_file, "r"))

    stat = {"meta": {"all_count": len(data)}}
    for item in data:
        types = item["question_type"].split("-")
        if types[0] not in stat.keys():
            stat[types[0]] = {}
        if types[1] not in stat[types[0]].keys():
            stat[types[0]][types[1]] = {"count": 0}
        stat[types[0]][types[1]]["count"] += 1

    return stat

def visualize(data, color_map, save_path):
    outer_categories = [k for k in data.keys() if k != "meta"]

    outer_counts, outer_labels = [], []
    for outer_key in outer_categories:
        outer_counts.append(sum(v['count'] for v in data[outer_key].values()))
        outer_labels.append(f'{outer_key}\n{outer_counts[-1]}')

    # 内圈
    inner_labels, inner_counts, inner_colors = [], [], []

    for category in outer_categories:
        subcategories = data[category]
        for subcat, subdata in subcategories.items():
            inner_labels.append(f"{subcat}\n{subdata['count']}")
            inner_counts.append(subdata['count'])
            inner_colors.append(color_map["inner"][category][subcat])
            # inner_colors.append(color_map[f"{category}:{subcat}"])

    # 外圈颜色（取大类第一个子类的颜色，保证一致性）
    # outer_colors = [color_map[f"{outer}:{list(data[outer].keys())[0]}"] for outer in outer_categories]
    outer_colors = [color_map["outer"][outer] for outer in outer_categories]
    # for category in outer_categories:
    #     for subcat, subdata in data[category].items():
    #         inner_colors.append(color_map["inner"][category][subcat])
    fig, ax = plt.subplots(figsize=(12, 10))

    wedges_outer, _ = ax.pie(outer_counts, radius=1.2,
        wedgeprops=dict(width=0.3, edgecolor='w'),
        colors=outer_colors, startangle=90)

    wedges_inner, _ = ax.pie(inner_counts, radius=0.9,
        wedgeprops=dict(width=0.3, edgecolor='w'),
        colors=inner_colors, startangle=90)

    # 添加标签
    for wedge, label in zip(wedges_outer, outer_labels):
        ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
        y = 1.0 * np.sin(np.deg2rad(ang))
        x = 1.0 * np.cos(np.deg2rad(ang))
        ax.text(x, y, label, ha="center", fontsize=15)

    for wedge, label in zip(wedges_inner, inner_labels):
        if label.split("\n")[0] in ["distance", "counting", "relationship", "status"]:
            continue
        ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
        y = 0.7 * np.sin(np.deg2rad(ang))
        x = 0.7 * np.cos(np.deg2rad(ang))
        ax.text(x, y, label, ha="center", fontsize=16)

    centre_circle = plt.Circle((0,0), 0.3, fc='white')
    ax.add_artist(centre_circle)

    plt.text(0, 0, f'Data Distribution\nTotal Count: {data["meta"]["all_count"]}',
             ha='center', va='center', fontsize=20)
    plt.savefig(save_path, bbox_inches='tight', transparent=True)


if __name__=="__main__":
    files = [
        "tmp/data/new_data/trainval.json",
        "tmp/data/new_data/seen_testset.json",
        "tmp/data/new_data/unseen_testset.json",
    ]
    stats = [statistic(f) for f in files]
    color_map = build_color_map(stats)

    for f, s in zip(files, stats):
        print(f"statistic {f}")
        out_path = f.replace(".json", "_distribution.pdf")
        visualize(s, color_map, out_path)
