import os
import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    """Dataset for bert fine-tuning"""
    def __init__(self, json_file, tokenizer, args):
        super(TextDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_path = args.data_path
        self.task = args.task
        self.from_json = json_file
        self.preprocess()

    def preprocess(self):
        self.preprocessed_data = []
        with open(self.from_json) as f:
            data_ori_list = json.load(f)
        for sample in data_ori_list:
            id = sample["id"]
            json_file = os.path.join(self.data_path, f"task-{self.task}", f"{id}-{self.task}.json")
            with open(json_file) as f:
                utt_list = json.load(f)
            text = "".join([utt["text"] for utt in utt_list])
            sample["input_ids"] = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt")["input_ids"]
            self.preprocessed_data.append(sample)

    def __getitem__(self, index):
        return self.preprocessed_data[index]
    
    def __len__(self):
        return len(self.preprocessed_data)
    

def collate_fn(batch):
    input_ids = pad_sequence([sample["input_ids"].squeeze() for sample in batch], batch_first=True, padding_value=0)
    attention_mask = (input_ids != 0)
    gender = torch.tensor([sample["gender"] for sample in batch])
    age = torch.tensor([sample["age"] for sample in batch])
    label = torch.tensor([sample["label"] for sample in batch])
    return {
        "input_ids": input_ids,
        "attention_masks": attention_mask,
        "genders": gender,
        "ages": age,
        "labels": label
    }
