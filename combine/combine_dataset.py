import json

import torch
from torch.utils.data import Dataset

class CombineDataset(Dataset):
    def __init__(self, json_file, speech_fea_dic, text_fea_dic):
        super(CombineDataset, self).__init__()
        self.from_json = json_file
        self.speech_fea_dic = speech_fea_dic
        self.text_fea_dic = text_fea_dic
        self.preprocess()

    def preprocess(self):
        self.preprocessed_data = []
        with open(self.from_json) as f:
            data_ori_list = json.load(f)
        for sample in data_ori_list:
            id = sample["id"]
            speech_fea = self.speech_fea_dic[id]
            text_fea = self.text_fea_dic[id]
            sample["speech_feature"] = torch.squeeze(speech_fea).cpu()
            sample["text_feature"] = torch.squeeze(text_fea).cpu()
            self.preprocessed_data.append(sample)

    def __getitem__(self, index):
        return self.preprocessed_data[index]
    
    def __len__(self):
        return len(self.preprocessed_data)
    

def collate_fn(batch):
    speech_feature = torch.stack([sample["speech_feature"] for sample in batch])
    text_feature = torch.stack([sample["text_feature"] for sample in batch])
    gender = torch.tensor([sample["gender"] for sample in batch])
    age = torch.tensor([sample["age"] for sample in batch])
    label = torch.tensor([sample["label"] for sample in batch])
    return {
        "speech_feature": speech_feature,
        "text_feature": text_feature,
        "genders": gender,
        "ages": age,
        "labels": label
    }
