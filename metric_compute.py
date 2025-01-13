import json
import sys
from sklearn.metrics import confusion_matrix
from audio.metric import metrics_compute

# label_json: Json file with groundtruth, obtained from preprocess/json_prepare.py
# pred_json: Json file with predictions
label_json = "data/test_data.json"

if len(sys.argv) > 1:
    pred_json = sys.argv[1]
else:
    pred_json = "src/sample.json"

with open(pred_json) as f:
    pred_dic = json.load(f)

with open(label_json) as f:
    label_list = json.load(f)
preds = []
refs = []
for sample in label_list:
    id = sample["id"]
    refs.append(sample["label"])
    preds.append(pred_dic[id])

with open(pred_json.replace("predictions.json", "metrics.txt"), 'a+') as f:
    f.write(label_json + '\n')
    f.write(str(metrics_compute(preds, refs, num_classes=2)))
    f.write('\n')
    f.write(str(confusion_matrix(refs, preds)))
    f.write('\n')