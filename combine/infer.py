import os
import json
import sys
import argparse

import numpy as np
import torch

from combine_model import ConcatModel

checkpoint = "best"
if len(sys.argv) > 1:
    ft_model_path = sys.argv[1]
else:
    ft_model_path = "results/task0/combine"
if len(sys.argv) == 3:
    checkpoint = sys.argv[2]
with open(os.path.join(ft_model_path, "model_config.json")) as f:
    train_config = json.load(f)
speech_file = train_config["speech_file"]
text_file = train_config["text_file"]
speech_fea_dic = np.load(speech_file, allow_pickle=True).item()
text_fea_dic = np.load(text_file, allow_pickle=True).item()

model = ConcatModel(argparse.Namespace(**train_config))
model.load_state_dict(torch.load(os.path.join(ft_model_path, f"{checkpoint}_model.pth")))
model.to("cuda")
model.eval()
print("Model loaded")

pred_dic = {}
logit_dic = {}
with torch.no_grad():
    for id in speech_fea_dic:
        speech_fea = torch.squeeze(speech_fea_dic[id]).to("cuda")
        text_fea = torch.squeeze(text_fea_dic[id]).to("cuda")
        logit = model(speech_fea, text_fea)
        pred = torch.argmax(logit, dim=-1).item()
        pred_dic[id] = pred
        logit_dic[id] = logit

with open(os.path.join(ft_model_path, f"{checkpoint}_predictions.json"), 'w') as f:
    json.dump(pred_dic, f, ensure_ascii=False, indent=4)
np.save(os.path.join(ft_model_path, f"{checkpoint}_logits.npy"), logit_dic)