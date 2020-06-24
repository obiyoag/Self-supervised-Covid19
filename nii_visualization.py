import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib


# 设置窗宽窗位，肺部一般是-600,1500
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


case_id = 220
slice_id = 12


# data_link = "Coronavirus_data/300cases/image/" + str(case_id).zfill(3) + ".nii.gz"
# lab_link = "Coronavirus_data/300cases/label/" + str(case_id).zfill(3) + ".nii.gz"


data_link = "nnUNet_data/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Covid19/imagesTr/" + str(case_id).zfill(3) + "_0000" + ".nii.gz"
lab_link = "nnUNet_data/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Covid19/labelsTr/"  + str(case_id).zfill(3) + ".nii.gz"


img_pros = nib.load(data_link)
img_lab = nib.load(lab_link)

pros_data = np.asarray(img_pros.dataobj)
pros_lab = np.asarray(img_lab.dataobj)

print(pros_data.shape)
print(pros_lab.shape)

# print(pros_lab.max())


# plt.figure(figsize=(10, 10))

# plt.subplot(121)
# plt.imshow((windowing(pros_data[:, :, slice_id])))

# plt.subplot(122)
# plt.imshow(pros_lab[:, :, slice_id])

# plt.show()
