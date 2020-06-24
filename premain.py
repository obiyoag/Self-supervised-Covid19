import torch
from dataSets import PreSet
from torchsummary import summary
from torch.optim import Adam, lr_scheduler
from utils import pre_train
from torch.utils.data import DataLoader
from models import UNet2D


batch_size = 16
epochs = 20
snr = 10
tv_weight = 1.
p_weight = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = PreSet("Coronavirus_data/300cases/dataset/pretrain", snr)

train_loader = DataLoader(trainset, batch_size, num_workers=8, pin_memory=True, shuffle=True)

params = {'in_chns':1,
              'feature_chns':[2, 8, 32, 48, 64],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 1,
              'bilinear': True}

UNet = UNet2D(params).to(device)

optimizer = Adam(UNet.parameters(), lr=0.001)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 4)

summary(UNet, (1, 256, 256))

for epoch in range(1, epochs + 1):
    pre_train(UNet, device, train_loader, optimizer, epoch, tv_weight, p_weight)
    scheduler.step()
torch.save(UNet, 'models2save/snr={:d}.pkl'.format(snr))

