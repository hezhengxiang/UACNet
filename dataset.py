import torch
from torch.utils.data import Dataset
import os
import glob
import pandas as pd
import numpy as np


def norm_data(x):
    data_min = min(x)
    data_max = max(x)
    normed_data = (x-data_min)/(data_max-data_min)
    return normed_data


class MyDataset(Dataset):
    def __init__(self, dataset, train=True):
        self.len_label_type = 0
        if train:
            self.train_dataset_dir = os.path.join(dataset, 'train')
        else:
            self.train_dataset_dir = os.path.join(dataset, 'test')
        self.data_index, self.labels = self.data_label_list()

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_index = self.data_index[idx]
        label = int(self.labels[idx])
        _data = np.loadtxt(file_index)
        _data = norm_data(_data)
        data = _data.reshape((1, -1))
        return data, label

    # def __getitem__(self, idx):
    #     file_index = self.data_index[idx]
    #     label_loc = self.labels[idx]
    #     label = np.zeros(self.len_label_type)
    #     _data = pd.read_csv(file_index, header=10, names=['x', 'y', 'z'])
    #     _data = _data[:]['x']
    #     data = np.array(_data).reshape((1, -1))
    #     label[int(label_loc)] = 1.
    #     return data, label

    def data_label_list(self):
        label_type = os.listdir(self.train_dataset_dir)
        self.len_label_type = len(label_type)
        file_list = []
        labels = []
        for label in label_type:
            file_dir = os.path.join(self.train_dataset_dir, label)
            _file_list = glob.glob(os.path.join(file_dir, '*.txt'))
            _labels = [label for i in range(len(_file_list))]
            file_list.extend(_file_list)
            labels.extend(_labels)
        return file_list, labels


