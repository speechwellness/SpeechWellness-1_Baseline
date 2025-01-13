import os
import json
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from audio.metric import metrics_compute

label_json_list = ["data/dev_data.json", "data/test_data.json"]

vote_list = [
    "results/task0/combine/lr1e-5",
    "results/task1/combine/lr1e-3",
    "results/task2/combine/lr5e-5",
]
output_dir = "results/soft_vote/combine"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

vote_pred_dic = {}

with open(os.path.join(output_dir, "logs.txt"), 'w') as f:
    f.write(str(vote_list) + '\n')

for pred_dir in vote_list:
    logit_dic = np.load(os.path.join(pred_dir, "best_logits.npy"), allow_pickle=True).item()
    for id in logit_dic:
        logit = logit_dic[id]
        prob = torch.softmax(logit, dim=-1).squeeze().unsqueeze(0).cpu()
        if id in vote_pred_dic:
            vote_pred_dic[id].append(prob)
        else:
            vote_pred_dic[id] = [prob]

for id in vote_pred_dic:
    probs = torch.concat(vote_pred_dic[id], dim=0)
    mean_prob = torch.mean(probs, dim=0)
    vote_pred_dic[id] = torch.argmax(mean_prob, dim=-1).item()

    
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