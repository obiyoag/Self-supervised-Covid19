import torch
from losses import DiceLoss


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    trainNum = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).view_as(target)
        loss = DiceLoss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx + 1) % 5 == 0:
            trainedNum = (batch_idx + 1) * len(data)
            print('Train Epoch: {} [{}/{}]\tDiceLoss: {:.6f}'.format(epoch, trainedNum, trainNum, loss))


def test(model, device, test_loader):
    model.eval()
    testNum = len(test_loader.dataset)
    diceloss = 0.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).view_as(target)
            diceloss += DiceLoss(output, target)

    diceloss /= testNum
    print('Test DiceLoss: {:.6f}'.format(diceloss))
