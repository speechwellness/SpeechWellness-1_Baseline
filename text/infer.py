import os
import json
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from text_model import TransformerWithHead

if len(sys.argv) > 1:
    ft_model_path = sys.argv[1]
else:
    ft_model_path = "results/task0/bert/lr2e-5"
with open(os.path.join(ft_model_path, "model_config.json")) as f:
    train_config = json.load(f)
pretrain_model_path = train_config["model_path"]
task = train_config["task"]
data_path = f"transcriptions/task-{task}"

tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
model = TransformerWithHead(pretrain_model_path)
model.load_state_dict(torch.load(os.path.join(ft_model_path, "best_model.pth")))
model.to("cuda")
model.eval()
print("Model loaded")

feature_dic = {}
pred_dic = {}
logits_dic = {}


with torch.no_grad():
    for json_file in os.listdir(data_path):
        id = json_file.split('-')[0]
        print(id)
        with open(os.path.join(data_path, json_file)) as f:
            utt_list = json.load(f)
        text = "".join([utt["text"] for utt in utt_list])
        input_ids = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")["input_ids"].to(model.transformer.device)
        logit = model(input_ids)
        prediction = torch.argmax(logit, dim=-1).item()
        feature = model.transformer(input_ids).last_hidden_state.squeeze().mean(dim=-2, keepdim=True)
        feature_dic[id] = feature
        pred_dic[id] = prediction
        logits_dic[id] = logit

np.save(os.path.join(ft_model_path, "embeddings.npy"), feature_dic)
np.save(os.path.join(ft_model_path, "logits.npy"), logits_dic)