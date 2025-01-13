import os
import json
import logging
import random
import numpy as np
from sklearn import svm, metrics
from sklearn.preprocessing import normalize

random.seed(0)

with open("data/train_data.json") as f:
    train_list = json.load(f)
with open("data/dev_data.json") as f:
    dev_list = json.load(f)
with open("data/test_data.json") as f:
    test_list = json.load(f)

random.shuffle(train_list)

task = '0'
result_dir = f"results/task{task}/SVM"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(filename=f'{result_dir}/SVM_log.log', level=logging.DEBUG) 
logging.info('Task: '+task)
feature_dic = np.load(f"eGeMAPS/eGeMAPS-{task}.npy", allow_pickle=True).item()

train_feature = np.stack([feature_dic[sample["id"]] for sample in train_list])
train_label = np.array([sample["label"] for sample in train_list])

dev_feature = np.stack([feature_dic[sample["id"]] for sample in dev_list])
dev_label = np.array([sample["label"] for sample in dev_list])
dev_id = [sample["id"] for sample in dev_list]

test_feature = np.stack([feature_dic[sample["id"]] for sample in test_list])
test_id = [sample["id"] for sample in test_list]

X_train_scaled = normalize(train_feature)
X_dev_scaled = normalize(dev_feature)
X_test_scaled = normalize(test_feature)

kernel = 'rbf'
c = 10
gamma = 100
model = svm.SVC(kernel='rbf', C=10, gamma=100)
model.fit(X_train_scaled, train_label)

dev_pred = model.predict(X_dev_scaled)
test_pred = model.predict(X_test_scaled)
acc = metrics.accuracy_score(y_true=dev_label, y_pred=dev_pred)
f1 = metrics.f1_score(y_true=dev_label, y_pred=dev_pred)
prec = metrics.precision_score(y_true=dev_label, y_pred=dev_pred)
recall = metrics.recall_score(y_true=dev_label, y_pred=dev_pred)
logging.info("Parameter, Kernel: "+str(kernel)+" C: "+str(c)+" gamma: "+str(gamma))
logging.info('The dev acc: '+str(acc)+" F1:"+str(f1)+" Precision:"+str(prec)+" Recall:"+str(recall))


dev_pred = model.predict(X_dev_scaled)
pred_dic = {}
for i, id in enumerate(dev_id):
    pred_dic[id] = dev_pred[i].item()
for i, id in enumerate(test_id):
    pred_dic[id] = test_pred[i].item()

with open(f"{result_dir}/predictions.json", 'w') as f:
    json.dump(pred_dic, f, indent=4)

