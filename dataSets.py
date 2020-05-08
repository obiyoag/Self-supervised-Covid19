import os
import torch
from torch.utils.data import Dataset
import numpy as np
# import matplotlib.pyplot as plt


def normalization_processing(data):

    data_mean = data.mean()
    data_std = data.std()

    data = data - data_mean
    data = data / data_std

    return data


# snr控制信噪比， 设置random_seed保证每次加噪声的随机性一样
def wgn(x, snr, random_seed):
    np.random.seed(random_seed)
    Ps = np.sum(abs(x)**2) / len(x)
    Pn = Ps / (10**((snr / 10)))
    noise = np.random.randn(len(x)) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise


class OriSet(Dataset):
    def __init__(self, path):
        self.data_path = path + "data/"
        self.lab_path = path + "label/"

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __getitem__(self, id):
        data = np.load(self.data_path + str(id) + ".npy").astype('float32')
        label = np.load(self.lab_path + str(id) + ".npy").astype('float32')
        data = normalization_processing(data)
        data = torch.tensor(data).unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0)
        return data, label


# oriset = OriSet("Covid19/Coronavirus_data/train/")
# data, label = oriset[604]
# data = data.unsqueeze(0)
# print(data.shape)
# plt.figure(figsize=(10, 10))

# plt.subplot(121)
# plt.imshow(data)

# plt.subplot(122)
# plt.imshow(label)

# plt.show()
