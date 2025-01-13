import os
import json
import math

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio

class AudioDataset(Dataset):
    """Dataset for W2V2 fine-tuning"""
    def __init__(self, json_file, feature_extractor, args):
        super(AudioDataset, self).__init__()
        self.feature_extractor = feature_extractor
        self.audio_path = args.audio_path
        self.segment_path = args.segment_path
        self.task = args.task
        self.remove_silence = args.remove_silence
        self.window_length_second = args.window_length
        self.step_length_second = args.step_length
        self.only_first_window = args.only_first_window
        self.from_json = json_file
        self.preprocess()

    def preprocess(self):
        self.preprocessed_data = []
        if self.task == "all":
            task_list = ['0', '1', '2']
        else:
            task_list = [self.task]
        with open(self.from_json) as f:
            data_ori_list = json.load(f)
        for task in task_list:
            for sample in data_ori_list:
                id = sample["id"]
                sample["task"] = int(task)
                audio_file = os.path.join(self.audio_path, f"task-{task}", f"{id}-{task}.wav")
                audio, sr = torchaudio.load(audio_file)
                assert sr == 16000
                if self.remove_silence:
                    json_file = os.path.join(self.segment_path, f"task-{task}", f"{id}-{task}.json")
                    with open(json_file) as f:
                        utt_list = json.load(f)
                    audio = torch.concat([audio[:, int(utt["start"]*16): int(utt["end"]*16)] for utt in utt_list], dim=1)
                
                audio_sec = audio.shape[1] / 16000
                window_num = math.ceil((audio_sec - (self.window_length_second - self.step_length_second)) / self.step_length_second) - 1

                # if only one window
                if window_num <= 0:
                    sample["input_values"] = self.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")["input_values"]
                    self.preprocessed_data.append(sample)
                    continue
                # more than one window
                if self.only_first_window:                
                    end = int(self.window_length_second * 16000)
                    sample["input_values"] = self.feature_extractor(audio[:, :end], sampling_rate=sr, return_tensors="pt")["input_values"]
                    self.preprocessed_data.append(sample)
                    continue
                for i in range(window_num-1):
                    start = int(i * self.step_length_second * 16000)
                    end = int((i * self.step_length_second + self.window_length_second) * 16000)
                    sample["input_values"] = self.feature_extractor(audio[:, start:end], sampling_rate=sr, return_tensors="pt")["input_values"]
                    self.preprocessed_data.append(sample)
                # Last window
                start = int(window_num * self.step_length_second * 16000)
                end = audio.shape[1]
                if (end - start) >= self.step_length_second * 16000:
                    sample["input_values"] = self.feature_extractor(audio[:, start:], sampling_rate=sr, return_tensors="pt")["input_values"]
                    self.preprocessed_data.append(sample)


    def __getitem__(self, index):
        return self.preprocessed_data[index]
    
    def __len__(self):
        return len(self.preprocessed_data)
    

def collate_fn(batch):
    input_values = pad_sequence([sample["input_values"].squeeze() for sample in batch], batch_first=True, padding_value=0)
    attention_mask = (input_values != 0)
    gender = torch.tensor([sample["gender"] for sample in batch])
    age = torch.tensor([sample["age"] for sample in batch])
    label = torch.tensor([sample["label"] for sample in batch])
    task = torch.tensor([sample["task"] for sample in batch])
    return {
        "input_values": input_values,
        "attention_masks": attention_mask,
        "genders": gender,
        "ages": age,
        "task": task,
        "labels": label
    }
