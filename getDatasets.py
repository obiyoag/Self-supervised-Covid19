import os
import numpy as np
from PIL import Image

train_indexes = [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14]
val_indexes = [6, 10, 15, 16, 17, 18, 19, 20]


def process(data, label, index):   # 对图像进行crop和resize操作以减少计算量
    if index <= 10:
        cropped_data, cropped_label = data[46:466, 46:466], label[46:466, 46:466]  # case1-10
    else:
        if index == 15:
            cropped_data, cropped_label = data[90:560, :], label[90:560, :]  # case15
        cropped_data, cropped_label = data[80:550, 80:550], label[80:550, 80:550]  # case11-14,16-20

    resized_data, resized_label = Image.fromarray(cropped_data).resize((256, 256)), Image.fromarray(cropped_label).resize((256, 256))
    data2save, label2save = np.array(resized_data), np.array(resized_label)
    return data2save, label2save


def getDataset(isTrain=True):
    nameFlag = 0
    if isTrain:
        indexes = train_indexes
        dataSavePath = "Covid19/Coronavirus_data/train/data/"
        labSavePath = "Covid19/Coronavirus_data/train/label/"
    else:
        indexes = val_indexes
        dataSavePath = "Covid19/Coronavirus_data/validation/data/"
        labSavePath = "Covid19/Coronavirus_data/validation/label/"

    if not (os.path.exists(dataSavePath) and os.path.exists(labSavePath)):
        os.makedirs(dataSavePath)
        os.makedirs(labSavePath)

    for index in indexes:
        data_path = "Covid19/Coronavirus_data/20cases/coronacases_" + str(index).zfill(3) + "/data/"
        lab_path = "Covid19/Coronavirus_data/20cases/coronacases_" + str(index).zfill(3) + "/label/"
        fileList = os.listdir(data_path)
        for file in fileList:
            data = np.load(data_path + file)
            label = np.load(lab_path + file)
            data, label = process(data, label, index)
            np.save(dataSavePath + str(nameFlag) + ".npy", data)
            np.save(labSavePath + str(nameFlag) + ".npy", label)
            nameFlag += 1


getDataset(True)
getDataset(False)
