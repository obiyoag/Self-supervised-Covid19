import torch
from dataSets import PreSet
from utils import pre_train, pre_test
from models import UNet1, UNet2D, UNet0
import matplotlib.pyplot as plt


batch_size = 16
epochs = 20
test_cases = [1, 6, 11, 15, 17]
tv_weight = 1.
p_weight = 0.8

snr = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset = PreSet("Coronavirus_data/dataset/train", snr)
dataset = PreSet("Coronavirus_data/300cases/dataset/pretask/preval", snr)

data, label = dataset[54]

label = label.squeeze()
noised_data = data.squeeze()

data = data.unsqueeze(0).to(device)
UNet = torch.load("models2save/tv_weight={:.1f}_pweight={:.1f}.pkl".format(tv_weight, p_weight))
# UNet1 = torch.load("models2save/snr={:d}_tvweight={:.1f}.pkl".format(snr, tvweight1))
# UNet2 = torch.load("models2save/snr={:d}_tvweight={:.1f}.pkl".format(snr, tvweight2))


# UNet = torch.load("models2save/snr={:d}.pkl".format(snr))
output = UNet(data)
output = output.squeeze().cpu().detach().numpy()

# output1 = UNet1(data)
# output1 = output1.squeeze().cpu().detach().numpy()

# output2 = UNet2(data)
# output2 = output2.squeeze().cpu().detach().numpy()


plt.figure(figsize=(10, 10))

plt.subplot(221)
plt.imshow(noised_data)

plt.subplot(222)
plt.imshow(label)

plt.subplot(223)
plt.imshow(output)

plt.show()

