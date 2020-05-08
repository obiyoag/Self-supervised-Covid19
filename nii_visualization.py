import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib


# 设置窗宽窗位，肺部一般是1500和-600
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


data_link = "Covid19/Coronavirus_data/20cases/coronacases_011.nii.gz"
lab_link = "Covid19/Coronavirus_data/inflection_mask/coronacases_011.nii.gz"


img_pros = nib.load(data_link)
img_lab = nib.load(lab_link)

pros_data = np.asarray(img_pros.dataobj)
pros_lab = np.asarray(img_lab.dataobj)

print(pros_data.shape)
print(pros_lab.shape)


plt.figure(figsize=(10, 10))

plt.subplot(121)
plt.imshow(pros_data[:, :, 23])

plt.subplot(122)
plt.imshow(windowing(pros_data[:, :, 23]))

plt.show()
