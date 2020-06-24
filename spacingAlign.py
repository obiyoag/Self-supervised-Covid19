import os
import SimpleITK as sitk


for case_id in range(290):
    data_link = "Coronavirus_data/300cases/image/" + str(case_id).zfill(3) + ".nii.gz"
    lab_link = "Coronavirus_data/300cases/label/" + str(case_id).zfill(3) + ".nii.gz"

    data_itk = sitk.ReadImage(data_link)
    label_itk = sitk.ReadImage(lab_link)

    assert data_itk.GetDirection() == label_itk.GetDirection()
    assert data_itk.GetOrigin() == label_itk.GetOrigin()
    assert data_itk.GetSize() == label_itk.GetSize()

    spacing = data_itk.GetSpacing()
    label_itk.SetSpacing(spacing)
    sitk.WriteImage(label_itk, lab_link)

