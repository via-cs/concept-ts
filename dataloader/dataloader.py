import os
import random
import numpy as np
import pandas as pd
import torch

from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


class LOF_Dataset(data.Dataset):
    def __init__(self, input_length, output_length, train, in_dim, data_path):
        self.in_dim = in_dim
        self.output_length = output_length
        self.input_length = input_length
        self.length = input_length + output_length
        self.data_path = data_path
        self.train = train

        # Generate output mask
        mask = [1] * self.length
        for i in range(self.output_length):
            mask[self.length - 1 - i] = 0
        self.mask = torch.Tensor(mask).reshape(self.length, 1)

        # Load data
        self.get_data(train, data_path)

    def __getitem__(self, index):
        start_id = index
        mid_id = start_id + self.length - self.output_length
        end_id = start_id + self.length

        # Build input feature + target mask
        input_feature = torch.Tensor(self.input[start_id:end_id])
        input_ans = torch.Tensor(self.output[start_id:end_id]) * self.mask
        input_data = torch.concat([input_feature, input_ans], dim=-1)

        # Ground truth for last output_length frames
        ans = torch.Tensor(self.output[mid_id:end_id])

        return (
            input_data.reshape(self.length, self.in_dim).detach(),
            ans.reshape(self.output_length, 1).detach(),
        )
        
    def get_data(self, train, data_path):
        dataset = pd.read_csv(data_path, encoding="utf-8")

        self.output = dataset["load"].values.reshape(-1, 1).tolist()
        dataset.drop("load", axis=1, inplace=True)
        self.input = dataset.values.tolist()

        if train in ["train", "test", "vali"]:
            train_end = int(0.7 * len(self.input))
            val_end = int(0.8 * len(self.input))
        else:
            train_end = len(self.input)
            val_end = None

        self.ans_type = StandardScaler()
        self.ans_type.fit_transform(self.output[:train_end])

        if train == "train":
            self.input = self.input[:train_end]
            self.output = self.output[:train_end]
        elif train == "vali":
            self.input = self.input[train_end:val_end]
            self.output = self.output[train_end:val_end]
        elif train == "test":
            self.input = self.input[val_end:]
            self.output = self.output[val_end:]

        self.output = self.ans_type.transform(self.output)

    def inver_data(self, inputs):
        return inputs * self.ans_type.scale_[0] + self.ans_type.mean_[0]

    def __len__(self):
        return len(self.input) - self.length + 1

    

    



                    
                    
