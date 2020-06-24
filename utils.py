import torch
from losses import DiceLoss, DenoiseLoss, log_loss
from dataSets import OriSet
from losses import DiceLoss
from torch.utils.data import DataLoader
import numpy as np
from torch.nn import MSELoss


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
        if(batch_idx + 1) % 20 == 0:
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
    dice = 1 - diceloss
    print('Test Dice: {:.6f}'.format(dice))
    return dice

def pre_train(model, device, train_loader, optimizer, epoch, tv_weight, p_weight):
    model.train()
    trainNum = len(train_loader.dataset)
    Criterion = DenoiseLoss(tv_weight, p_weight)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).view_as(target)
        loss = Criterion(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx + 1) % 20 == 0:
            trainedNum = (batch_idx + 1) * len(data)
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, trainedNum, trainNum, loss))

    
class CaseTester():
    
    def __init__(self, model, device, test_cases):
        self.model = model
        self.device = device
        self.test_cases = test_cases
        self.test_result = []
        

    def get_loader(self, case_id):
        case_path = "Coronavirus_data/300cases/case_data/coronacases_" + str(case_id).zfill(3)
        case_set = OriSet(case_path)
        case_loader = DataLoader(case_set, 1, num_workers=0, pin_memory=False, shuffle=True)
        return case_loader
        
    def case_test(self, model, device, test_loader, case_id):
        self.model.eval()
        testNum = len(test_loader.dataset)
        diceloss = 0.
        Criterion = DiceLoss()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).view_as(target)
                diceloss += Criterion(output, target)

        diceloss /= testNum
        dice = 1 - diceloss
        return dice.item()

    def run_test(self):
        for case_id in self.test_cases:
            case_loader = self.get_loader(case_id)
            case_dice = self.case_test(self.model, self.device, case_loader, case_id)
            self.test_result.append(case_dice)
        aveDice = np.array(self.test_result).mean()
        self.test_result = [] # 清空test_result
        print("The average test Dice is {:.4f}".format(aveDice))
        return aveDice
