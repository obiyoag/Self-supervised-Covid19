import matplotlib.pyplot as plt
import numpy as np
from dataSets import PreSet


def wgn2D(x, snr, random_seed=40):
    np.random.seed(random_seed)
    Ps = np.sum(abs(x)**2) / np.size(x)
    Pn = Ps / (10**((snr / 10)))
    noise = np.random.randn(x.shape[0], x.shape[1]) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise

case_id = 54
slice_id = 12000

data_link = "Coronavirus_data/300cases/dataset/pretask/pretrain/data/" + str(slice_id) + ".npy"
lab_link = "Coronavirus_data/300cases/dataset/pretask/pretrain/label/" + str(slice_id) + ".npy"


data = np.load(data_link)/255
label = np.load(lab_link)
noised_data = wgn2D(data, 15, 4)


plt.figure(figsize=(10, 10))

plt.subplot(121)
plt.imshow(data)

plt.subplot(122)
plt.imshow(label)

plt.show()
