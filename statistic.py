import numpy as np
import nibabel as nib
import pandas as pd

#  统计带病灶的slice个数
lesionArr = np.zeros(420)

for case_id in range(1, 21):
    lab_link = "Covid19/Coronavirus_data/inflection_mask/coronacases_" + str(case_id).zfill(3) + ".nii.gz"
    img_lab = nib.load(lab_link)
    pros_lab = np.asarray(img_lab.dataobj)
    count = 0

    lesionList = np.zeros(420)
    for channel_id in range(0, pros_lab.shape[2]):

        if pros_lab.max() == 3:  # 对于后10个cases
            if pros_lab[:, :, channel_id].max() == 3:
                lesionList[channel_id] += 1
                count += 1

        if pros_lab.max() == 1:  # 对于前10个cases
            if pros_lab[:, :, channel_id].max() == 1:
                lesionList[channel_id] += 1
                count += 1

    lesionList = np.where(lesionList == 1)[0]
    lesionList = np.pad(lesionList, (0, 420 - lesionList.shape[0]))
    lesionList[249] = count
    lesionArr = np.vstack((lesionArr, lesionList))
    print(lesionArr)

lesionArr = lesionArr[1:, :250]
df = pd.DataFrame(lesionArr.astype(np.int32))
df.to_csv("Covid19/Coronavirus_data/lesionArr.csv")
