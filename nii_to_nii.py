import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
from PIL import Image



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

# 设置窗宽窗位，是-600,1500
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


for case_id in range(1, 21):
    data_link = "Covid19/Coronavirus_data/20cases/coronacases_" + str(case_id).zfill(3) + ".nii.gz"
    lab_link = "Covid19/Coronavirus_data/inflection_mask/coronacases_" + str(case_id).zfill(3) + ".nii.gz"
    data_path = "Covid19/nnUNet_data/nii_data/data/COVID_" + str(case_id).zfill(3) + "_0000.nii.gz"
    lab_path = "Covid19/nnUNet_data/nii_data/label/COVID_" + str(case_id).zfill(3) + ".nii.gz"

    img_pros = nib.load(data_link)
    img_lab = nib.load(lab_link)
    pros_data = np.asarray(img_pros.dataobj)
    pros_lab = np.asarray(img_lab.dataobj)

    slice_num = pros_data.shape[2]

    case_data = np.zeros((256,256,slice_num))
    case_label = np.zeros((256,256,slice_num))

    for index in range(slice_num):

        if case_id <= 10:  # 前10个cases需要设置窗宽窗位
            data = windowing(pros_data[:, :, index])
        else:  # 后10个cases不需要设置
            data = pros_data[:, :, index]

        label = pros_lab[:, :, index]

        data, label = process(data, label, case_id)

        case_data[:,:,index] = data
        case_label[:,:,index] = label

    
    
    case_data = nib.Nifti1Image(case_data, np.eye(4)) # np.eye是仿射变换,需要加上
    case_label = nib.Nifti1Image(case_label, np.eye(4))

    nib.save(case_data, data_path)
    nib.save(case_label, lab_path)
