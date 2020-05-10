import torch
from losses import DiceLoss


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    trainNum = len(train_loader.dataset)
    Criterion = DiceLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).view_as(target)
        loss = Criterion(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx + 1) % 5 == 0:
            trainedNum = (batch_idx + 1) * len(data)
            print('Train Epoch: {} [{}/{}]\tDiceLoss: {:.6f}'.format(epoch, trainedNum, trainNum, loss))


def test(model, device, test_loader):
    model.eval()
    testBatchNum = len(test_loader)
    diceloss = 0.
    Criterion = DiceLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).view_as(target)
            diceloss += Criterion(output, target)

    diceloss /= testBatchNum
    print('Test DiceLoss: {:.6f}'.format(diceloss))
