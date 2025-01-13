import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix
from audio.metric import metrics_compute

label_json_list = ["data/dev_data.json", "data/test_data.json"]

vote_list = [
    "results/task0/SVM",
    "results/task1/SVM",
    "results/task2/SVM",
]

output_dir = "results/vote/SVM"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

vote_pred_dic = {}

with open(os.path.join(output_dir, "logs.txt"), 'w') as f:
    f.write(str(vote_list) + '\n')

for pred_json in vote_list:
    with open(os.path.join(pred_json, "predictions.json")) as f:
        pred_dic = json.load(f)
    for id in pred_dic:
        if id in vote_pred_dic:
            vote_pred_dic[id].append(pred_dic[id])
        else:
            vote_pred_dic[id] = [pred_dic[id]]

for id in vote_pred_dic:
    vote_pred_dic[id] = 1 if np.mean(vote_pred_dic[id]) >= 0.5 else 0

with open(os.path.join(output_dir, "vote_result.json"), 'w') as f:
    json.dump(vote_pred_dic, f, indent=4)

for label_json in label_json_list:
    with open(label_json) as f:
        label_list = json.load(f)
    preds = []
    refs = []
    for sample in label_list:
        id = sample["id"]
        refs.append(sample["label"])
        preds.append(vote_pred_dic[id])

    with open(os.path.join(output_dir, "logs.txt"), 'a+') as f:
        f.write(label_json + '\n')
        f.write(str(metrics_compute(preds, refs, num_classes=2)))
        f.write('\n')
        f.write(str(confusion_matrix(refs, preds)))
        f.write('\n')
    print(str(metrics_compute(preds, refs, num_classes=2)))