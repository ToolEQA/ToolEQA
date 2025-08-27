import json
import random

def statistic_scenes(data_file):
    # 统计每个场景包含多少sample_id
    data = json.load(open(data_file, "r"))
    scene_dict = {}
    for item in data:
        scene = item["scene"]
        if scene not in scene_dict.keys():
            scene_dict[scene] = []
        scene_dict[scene].append(item)

    return scene_dict

if __name__=="__main__":
    data_file = "data/ToolTrajectory/full.json"
    scene_dict = statistic_scenes(data_file)
    print("一共 {} 个场景".format(len(scene_dict.keys())))

    seen_set = []
    unseen_set = []
    trainval_set = []

    remain_set = []

    for k, v in scene_dict.items():
        if len(v) < 12 and len(v) > 4:
            # for task in v:
            #     type_count.add(task["question_type"])
            # scene_count += 1
            # task_count += len(v)
            unseen_set.extend(v)
        else:
            remain_set.extend(v)
            
    seen_set = random.sample(remain_set, int(len(remain_set) * 0.05))

    sampled_ids = {item['sample_id'] for item in seen_set}
    trainval_set = [item for item in remain_set if item['sample_id'] not in sampled_ids]


    cate = set()
    scene = set()
    for item in unseen_set:
        cate.add(item["question_type"])
        scene.add(item["scene"])
    print("============= unseen test set =============")
    print("包含的类别有: ", len(cate))
    print("涉及的场景数量: ", len(scene))
    print("任务数: ", len(unseen_set))
    with open("unseen_testset.json", "w") as f:
        json.dump(unseen_set, f, indent=4)


    cate = set()
    scene = set()
    for item in seen_set:
        cate.add(item["question_type"])
        scene.add(item["scene"])
    print("============= seen test set =============")
    print("包含的类别有: ", len(cate))
    print("涉及的场景数量: ", len(scene))
    print("任务数: ", len(seen_set))
    with open("seen_testset.json", "w") as f:
        json.dump(seen_set, f, indent=4)

    cate = set()
    scene = set()
    for item in trainval_set:
        cate.add(item["question_type"])
        scene.add(item["scene"])
    print("============= trainval set =============")
    print("包含的类别有: ", len(cate))
    print("涉及的场景数量: ", len(scene))
    print("任务数: ", len(trainval_set))
    with open("trainval.json", "w") as f:
        json.dump(trainval_set, f, indent=4)