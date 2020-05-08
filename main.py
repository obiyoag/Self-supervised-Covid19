import torch
from dataSets import OriSet
from torchsummary import summary
from torch.optim import Adam
from utils import train, test
from torch.utils.data import DataLoader
from models import UNet0


batch_size = 8
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = OriSet("Covid19/Coronavirus_data/train/")
testset = OriSet("Covid19/Coronavirus_data/validation/")

train_loader = DataLoader(trainset, batch_size, num_workers=8, pin_memory=True)
test_loader = DataLoader(testset, batch_size, num_workers=8, pin_memory=True)

UNet = UNet0().to(device)
optimizer = Adam(UNet.parameters(), lr=1e-4)

summary(UNet, (1, 256, 256))

for epoch in range(1, epochs + 1):
    train(UNet, device, train_loader, optimizer, epoch)
    test(UNet, device, test_loader)
