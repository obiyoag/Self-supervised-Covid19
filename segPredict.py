import torch
from dataSets import OriSet
from utils import pre_train, pre_test
from models import UNet1, UNet2D, UNet0
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = OriSet("Coronavirus_data/dataset/validation")

data, label = dataset[512]

label = label.squeeze()
noised_data = data.squeeze()

data = data.unsqueeze(0).to(device)
UNet = torch.load("models2save/test_visual.pkl")
output = UNet(data)
output = output.squeeze().cpu().detach().numpy()


plt.figure(figsize=(10, 10))

plt.subplot(121)
plt.imshow(label)

plt.subplot(122)
plt.imshow(output)

plt.show()

