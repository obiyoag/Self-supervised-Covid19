import os
import numpy as np
import nibabel as nib
from PIL import Image


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

def process(data, label, index):   
    resized_data, resized_label = Image.fromarray(data).resize((256, 256)), Image.fromarray(label).resize((256, 256))
    data2save, label2save = np.array(resized_data), np.array(resized_label)
    return data2save, label2save

for case_id in range(290):
    data_link = "Coronavirus_data/300cases/image/" + str(case_id).zfill(3) + ".nii.gz"
    lab_link = "Coronavirus_data/300cases/label/" + str(case_id).zfill(3) + ".nii.gz"
    data_path = "Coronavirus_data/300cases/case_data/coronacases_" + str(case_id).zfill(3) + "/data/"
    lab_path = "Coronavirus_data/300cases/case_data/coronacases_" + str(case_id).zfill(3) + "/label/"

    img_pros = nib.load(data_link)
    img_lab = nib.load(lab_link)
    pros_data = np.asarray(img_pros.dataobj)
    pros_lab = np.asarray(img_lab.dataobj)


    indexes = range(pros_data.shape[2])
    slice_id = 0
    for index in indexes:

        data = windowing(pros_data[:, :, index], True)
        label = pros_lab[:, :, index]

        data, label = process(data, label, index)

        if not (os.path.exists(data_path) and os.path.exists(lab_path)):
            os.makedirs(data_path)
            os.makedirs(lab_path)

        np.save(data_path + str(slice_id) + ".npy", data)
        np.save(lab_path + str(slice_id) + ".npy", label)
        slice_id += 1
    
    print("Case {:d} is done.".format(case_id))
