import torch
from dataSets import OriSet
from torchsummary import summary
from torch.optim import Adam, lr_scheduler, SGD
from utils import train, CaseTester
from torch.utils.data import DataLoader
from models import UNet2D


batch_size = 28
epochs = 150
test_cases = range(240, 290)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frozen_layers = 0
pretrain = True
snr = 10

trainset = OriSet("Coronavirus_data/300cases/dataset/train_40cases")
train_loader = DataLoader(trainset, batch_size, num_workers=0, pin_memory=False, shuffle=True)

params = {'in_chns':1, 'feature_chns':[2, 8, 32, 48, 64], 'dropout':  [0, 0, 0.3, 0.4, 0.5], 'class_num': 1, 'bilinear': True}

UNet = UNet2D(params).to(device)

if pretrain is True:
    # 载入预训练模型
    UNet = torch.load("models2save/snr={:d}.pkl".format(snr))
    # 设置一些层不参与更新
    ft_flag = 0
    for child in UNet.children():
        if ft_flag < frozen_layers:
            for param in child.parameters():
                param.requires_grad = False
        ft_flag += 1


# lr=5e-3
# inconv_params = list(map(id, UNet.in_conv.parameters()))
# down1_params = list(map(id, UNet.down1.parameters()))
# down2_params = list(map(id, UNet.down2.parameters()))
# down3_params = list(map(id, UNet.down3.parameters()))
# down4_params = list(map(id, UNet.down4.parameters()))


# up_params = filter(lambda p: id(p) not in inconv_params + down1_params + down2_params + down3_params + down4_params, UNet.parameters())
# optimizer = torch.optim.Adam([
#             {'params': UNet.in_conv.parameters(), 'lr': lr * 1e-1},
#             {'params': UNet.down1.parameters(), 'lr': lr * 1e-1},
#             {'params': UNet.down2.parameters(), 'lr': lr * 1e-1},
#             {'params': UNet.down3.parameters(), 'lr': lr * 0.5},
#             {'params': UNet.down4.parameters(), 'lr': lr * 0.5},
#             {'params': up_params}], lr=lr)

optimizer = Adam(UNet.parameters(), lr=0.005)
# optimizer = SGD(UNet.parameters(), lr=1e-3, momentum=0.5, weight_decay=1e-4)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 4)
tester = CaseTester(UNet, device, test_cases)

summary(UNet, (1, 256, 256))

bestDice = 0.
for epoch in range(1, epochs + 1):
    train(UNet, device, train_loader, optimizer, epoch)
    scheduler.step()
    currentDice =  tester.run_test()
    if currentDice > bestDice:
        torch.save(UNet, 'models2save/best_model1.pkl')
        bestDice = currentDice
        print(bestDice.item())
print(bestDice.item())

