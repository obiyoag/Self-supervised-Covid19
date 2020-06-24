import os
import numpy as np


pretrain_index = range(220)
train_index = range(180, 220)
val_index = range(220, 240)
test_index = range(240, 290)

def getDataset(dataset):
    nameFlag = 0
    if dataset == "pretrain":
        indexes = pretrain_index
        dataSavePath = "Coronavirus_data/300cases/dataset/pretrain/data/"
        labSavePath = "Coronavirus_data/300cases/dataset/pretrain/label/"
    if dataset == "train":
        indexes = train_index
        dataSavePath = "Coronavirus_data/300cases/dataset/train_40cases/data/"
        labSavePath = "Coronavirus_data/300cases/dataset/train_40cases/label/"
    if dataset == "val":
        indexes = val_index
        dataSavePath = "Coronavirus_data/300cases/dataset/val/data/"
        labSavePath = "Coronavirus_data/300cases/dataset/val/label/"
    if dataset == "test":
        indexes = test_index
        dataSavePath = "Coronavirus_data/300cases/dataset/test/data/"
        labSavePath = "Coronavirus_data/300cases/dataset/test/label/"

    if not (os.path.exists(dataSavePath) and os.path.exists(labSavePath)):
        os.makedirs(dataSavePath)
        os.makedirs(labSavePath)

    for index in indexes:
        data_path = "Coronavirus_data/300cases/case_data/coronacases_" + str(index).zfill(3) + "/data/"
        lab_path = "Coronavirus_data/300cases/case_data/coronacases_" + str(index).zfill(3) + "/label/"
        file_num = len(os.listdir(data_path))
        saved_num = len(os.listdir(dataSavePath))
        for file_id in range(file_num):
            data = np.load(data_path + str(file_id) + ".npy")
            label = np.load(lab_path + str(file_id) + ".npy")
            np.save(dataSavePath + str(nameFlag) + ".npy", data)
            np.save(labSavePath + str(nameFlag) + ".npy", label)
            nameFlag += 1

        print("case " + str(index) + " finished")

# getDataset("pretrain")
getDataset("train")
# getDataset("val")
# getDataset("test")
