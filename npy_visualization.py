import matplotlib.pyplot as plt
import numpy as np

data_link = "Covid19/Coronavirus_data/train/data/981.npy"
lab_link = "Covid19/Coronavirus_data/train/label/981.npy"

data = np.load(data_link)
label = np.load(lab_link)


plt.figure(figsize=(10, 10))

plt.subplot(121)
plt.imshow(data)

plt.subplot(122)
plt.imshow(label)

plt.show()
