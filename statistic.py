import os
import nibabel as nib
import numpy as np

# rename
# data_link = "Coronavirus_data/300cases/image/"
# lab_link = "Coronavirus_data/300cases/label/"

# data_list = os.listdir(data_link)
# lab_list = os.listdir(lab_link)

# for id in range(len(data_list)):
#     os.rename(data_link + data_list[id], data_link + str(id).zfill(3) + ".nii.gz")
#     os.rename(lab_link + lab_list[id], lab_link + str(id).zfill(3) + ".nii.gz")


data_link = "nnUNet_data/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Covid19/imagesTr/"
lab_link = "nnUNet_data/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Covid19/labelsTr/"

for id in range(200, 220):
    os.rename(data_link + str(id).zfill(3) + ".nii.gz", data_link + str(id).zfill(3)+ "_0000" + ".nii.gz")
    # os.rename(lab_link + lab_list[id], lab_link + str(id+220).zfill(3) + ".nii.gz")


# for id in range(300):
#     data_link = "Coronavirus_data/300cases/image/" + str(id).zfill(3) + ".nii.gz"
#     lab_link = "Coronavirus_data/300cases/label/" + str(id).zfill(3) + ".nii.gz"

#     img_pros = nib.load(data_link)
#     img_lab = nib.load(lab_link)

#     pros_data = np.asarray(img_pros.dataobj)
#     pros_lab = np.asarray(img_lab.dataobj)

    # print(pros_data.shape)