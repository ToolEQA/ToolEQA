import pandas as pd
import os
import numpy as np
import json

from matplotlib import pyplot as plt
from collections import defaultdict

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

def visualize(data):
    # 准备外圈（大类）数据
    outer_categories = list(data.keys())
    outer_categories.remove('meta')  # 移除元数据

    outer_counts = []
    for outer_key in outer_categories:
        outer_counts.append(sum(v['count'] for v in data[outer_key].values()))
        
    outer_labels = [f'{cat}\n{count}' for cat, count in zip(outer_categories, outer_counts)]

    # 准备内圈（小类）数据
    inner_labels = []
    inner_counts = []
    inner_colors = []

    # 为每个大类分配一个颜色，小类使用该颜色的不同深浅
    colors = plt.cm.tab20.colors

    for i, category in enumerate(outer_categories):
        if category in ['attribute', 'location']:  # 有子类的大类
            subcategories = data[category]
            for j, (subcat, subdata) in enumerate(subcategories.items()):
                inner_labels.append(f"{subcat}\n{subdata['count']}")
                inner_counts.append(subdata['count'])
                # 使用大类的颜色但调整亮度
                inner_colors.append(colors[i*2 + j%2])
        else:  # 没有子类的大类
            inner_labels.append(f"{category}\n{data[category][category]['count']}")
            inner_counts.append(data[category][category]['count'])
            inner_colors.append(colors[i*2])

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 10))

    # 外圈饼图（标签显示在内部）
    wedges_outer, _ = ax.pie(outer_counts, radius=1.2,
        wedgeprops=dict(width=0.3, edgecolor='w'),
        colors=colors[::2],
        startangle=90)

    # 内圈饼图（标签显示在内部）
    wedges_inner, _ = ax.pie(inner_counts, radius=0.9,
        wedgeprops=dict(width=0.3, edgecolor='w'),
        colors=inner_colors,
        startangle=90)

    # 添加标签到饼图内部
    # bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="w", lw=0.5, alpha=0.8)

    for wedge, label in zip(wedges_outer, outer_labels):
        ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
        y = 1.0 * np.sin(np.deg2rad(ang))
        x = 1.0 * np.cos(np.deg2rad(ang))
        horizontalalignment = "center"
        ax.text(x, y, label, horizontalalignment=horizontalalignment,
                fontsize=11)

    for wedge, label in zip(wedges_inner, inner_labels):
        ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
        y = 0.7 * np.sin(np.deg2rad(ang))
        x = 0.7 * np.cos(np.deg2rad(ang))
        horizontalalignment = "center"
        ax.text(x, y, label, horizontalalignment=horizontalalignment,
                fontsize=9)

    # 添加中心空白
    centre_circle = plt.Circle((0,0), 0.3, fc='white')
    ax.add_artist(centre_circle)

    plt.text(0, 0, f'Data Distribution\nTotal Count: {data["meta"]["all_count"]}', ha='center', va='center', fontsize=16)
    plt.savefig("./data/ToolTrajectory/data_distribution.png", dpi=300, bbox_inches='tight')


if __name__=="__main__":
    data_file = "/mynvme1/EQA-Traj-0715/trajectory.json"
    data = statistic(data_file)
    print(data)
    visualize(data)