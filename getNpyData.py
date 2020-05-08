import os
import numpy as np
import pandas as pd
import nibabel as nib


def windowing(image, normal=False):
    window_center = -600
    window_width = 1500
    minWindow = float(window_center) - 0.5 * float(window_width)
    newimg = (image - minWindow) / float(window_width)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


lesionArr = pd.read_csv("Covid19/Coronavirus_data/lesionArr.csv").values[:, 1:]
for case_id in range(1, 21):
    data_link = "Covid19/Coronavirus_data/20cases/coronacases_" + str(case_id).zfill(3) + ".nii.gz"
    lab_link = "Covid19/Coronavirus_data/inflection_mask/coronacases_" + str(case_id).zfill(3) + ".nii.gz"
    data_path = "Covid19/Coronavirus_data/20cases/coronacases_" + str(case_id).zfill(3) + "/data/"
    lab_path = "Covid19/Coronavirus_data/20cases/coronacases_" + str(case_id).zfill(3) + "/label/"

    img_pros = nib.load(data_link)
    img_lab = nib.load(lab_link)
    pros_data = np.asarray(img_pros.dataobj)
    pros_lab = np.asarray(img_lab.dataobj)
    print(pros_data.shape)

    indexes = lesionArr[case_id - 1, :]
    slice_id = 0
    for index in indexes:

        if index == 0:  # 读取到0意味着读取结束,跳出循环
            break
        if index == 500:  # 为避免第一个0读取不上，将第一个0记作500以区分
            index = 0

        if case_id <= 10:  # 前10个cases需要设置窗宽窗位
            data = windowing(pros_data[:, :, index])
        else:  # 后10个cases不需要设置
            data = pros_data[:, :, index]
        label = pros_lab[:, :, index]

        if not (os.path.exists(data_path) and os.path.exists(lab_path)):
            os.makedirs(data_path)
            os.makedirs(lab_path)

        np.save(data_path + str(slice_id) + ".npy", data)
        np.save(lab_path + str(slice_id) + ".npy", label)
        slice_id += 1
