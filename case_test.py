import torch
from dataSets import OriSet
from losses import DiceLoss
from torch.utils.data import DataLoader
import numpy as np


def case_test(model, device, test_loader, case_id):
    model.eval()
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
    print('Test Dice of case {}: {:.6f}'.format(case_id, dice))
    return dice.item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNet = torch.load('models2save/best_model.pkl')


test_cases = [1, 6, 11, 15, 17]
test_results = []
for case_id in test_cases:
    case_path = "Coronavirus_data/case_processed/coronacases_" + str(case_id).zfill(3)
    test_set = OriSet(case_path)
    test_loader = DataLoader(test_set, 1, num_workers=8, pin_memory=True, shuffle=True)
    case_dice = case_test(UNet, device, test_loader, case_id)
    test_results.append(case_dice)

print("The average testi Dice is {:.4f}".format(np.array(test_results).mean()))



    

