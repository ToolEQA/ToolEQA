import pandas as pd
import os

def statistic(root):
    stat = {"meta": {"all_count": 0}}
    for quest_type in os.listdir(root):
        stat[quest_type] = {}
        for quest_csv in os.listdir(os.path.join(root, quest_type)):
            data = pd.read_csv(os.path.join(root, quest_type, quest_csv))
            sub_type = quest_csv.split(".")[0]
            stat[quest_type][sub_type] = {}
            stat[quest_type][sub_type]["count"] = len(data)
            stat["meta"]["all_count"] += len(data)
    print(stat)

if __name__=="__main__":
    root = "data/ToolTrajectory/questions"
    statistic(root)