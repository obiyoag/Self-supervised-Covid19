import os
import torch
from torch.utils.data import Dataset
import numpy as np


def normalization_processing(data):

    data /= 256
    data_mean = data.mean()
    data_std = data.std()

    data = data - data_mean
    data = data / data_std

    return data


# snr控制信噪比， 设置random_seed保证每次加噪声的随机性一样
def wgn2D(x, snr, random_seed=40):
    np.random.seed(random_seed)
    Ps = np.sum(abs(x)**2) / np.size(x)
    Pn = Ps / (10**((snr / 10)))
    noise = np.random.randn(x.shape[0], x.shape[1]) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise


class OriSet(Dataset):
    def __init__(self, path):
        self.data_path = path + "/data/"
        self.lab_path = path + "/label/"

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __getitem__(self, id):
        data = np.load(self.data_path + str(id) + ".npy").astype('float32')
        label = np.load(self.lab_path + str(id) + ".npy").astype('float32')
        data = normalization_processing(data)
        data = torch.FloatTensor(data).unsqueeze(0)
        label = torch.FloatTensor(label).unsqueeze(0)
        return data, label

class PreSet(Dataset):
    def __init__(self, path, snr):
        self.data_path = path + "/data/"
        self.snr = snr

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __getitem__(self, id):
        label = np.load(self.data_path + str(id) + ".npy").astype('float32') / 256
        label = normalization_processing(label)
        data = wgn2D(label.copy(), self.snr, id + 1)
        label = torch.FloatTensor(label).unsqueeze(0)
        data = torch.FloatTensor(data).unsqueeze(0)
        return data, label



