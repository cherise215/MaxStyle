import os

import numpy as np
import glob
import SimpleITK as sitk
from os.path import join
import matplotlib.pyplot as plt
import SimpleITK as sitk
import sys
import time

sys.path.append('../')

from src.dataset_loader.dataset_utils import resample_by_spacing


def normalize_minmax_data(image_data):
    """
    # 3D MRI scan is normalized to range between 0 and 1 using min-max normalization.
    Here, the minimum and maximum values are used as 2nd and 98th percentiles respectively from the 3D MRI scan.
    We expect the outliers to be away from the range of [0,1].
    input params :
        image_data : 3D MRI scan to be normalized using min-max normalization
    returns:
        final_image_data : Normalized 3D MRI scan obtained via min-max normalization.
    """
    min_val_2p = np.percentile(image_data, 2)
    max_val_98p = np.percentile(image_data, 98)
    final_image_data = np.zeros(
        (image_data.shape[0], image_data.shape[1], image_data.shape[2]), dtype=np.float32)
    # min-max norm on total 3D volume
    image_data[image_data < min_val_2p] = min_val_2p
    image_data[image_data > max_val_98p] = max_val_98p

    final_image_data = (image_data - min_val_2p) / (1e-10 + max_val_98p - min_val_2p)

    return final_image_data


def crop_or_pad_slice_to_size(img_slice, nx, ny):
    """
    To crop the input 2D slice for the given dimensions
    input params :
        image_slice : 2D slice to be cropped
        nx : dimension in x
        ny : dimension in y
    returns:
        slice_cropped : cropped 2D slice
    """
    slice_cropped = np.zeros((nx, ny))
    x, y = img_slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = img_slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = img_slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = img_slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = img_slice[:, :]

    return slice_cropped


def correct_image(sitkImage, threshhold=0.001):
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    start = time.time()

    maskImage = sitkImage > threshhold
    correctedImage = corrector.Execute(sitkImage, maskImage)
    end = time.time()

    #print ('debiasing costs {} s'.format(end-start))
    return correctedImage, end - start


def resample_np_array(normalized_array, old_spacing, interp=sitk.sitkLinear, keep_z_spacing=True, new_spacing=[1.367, 1.367, -1]):
    sitkImage = sitk.GetImageFromArray(normalized_array)
    sitkImage.SetSpacing(spacing=old_spacing)
    resampleImage = resample_by_spacing(
        sitkImage, new_spacing, keep_z_spacing=keep_z_spacing, interpolator=interp)
    resampleArray = sitk.GetArrayFromImage(resampleImage)
    new_spacing = resampleImage.GetSpacing()
    ##print (new_spacing)
    return resampleArray


if __name__ == '__main__':
    total_bias_correct_time = 0.
    count = 0
    new_spacing = [1.36719, 1.36719, -1]
    pid_list = ['%03d' % (i + 1) for i in range(29, 100)]

    for pid in pid_list:
        for frame in ['ED', 'ES']:
            image_path_format = "/vol/medic02/users/cc215/data/ACDC/dataset/all/patient{}/image_" + frame + ".nii.gz"
            label_path_format = "/vol/medic02/users/cc215/data/ACDC/dataset/all/patient{}/label_" + frame + ".nii.gz"
            output_dir = '/vol/medic02/users/cc215/data/ACDC/dataset/preprocessed/' + frame
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            count += 1
            print(pid)
            # load image and label
            sitkImage = sitk.ReadImage(image_path_format.format(pid))
            sitkImage = sitk.Cast(sitkImage, sitk.sitkFloat32)
            sitkLabel = sitk.ReadImage(label_path_format.format(pid))
            sitkLabel = sitk.Cast(sitkLabel, sitk.sitkInt16)

            orig_spacing = sitkImage.GetSpacing()

            # correct bias on images
            #sitkImage, cost_time = correct_image(sitkImage, threshhold=0.001)
            #total_bias_correct_time += cost_time
            print(sitkImage.GetDirection())

            # intensity normalization on images
            imgArray = sitk.GetArrayFromImage(sitkImage)
            normalized_array = normalize_minmax_data(imgArray)

            # resample image and label
            resampled_image_array = resample_np_array(
                normalized_array, old_spacing=orig_spacing, interp=sitk.sitkLinear, keep_z_spacing=True, new_spacing=new_spacing)

            label_array = sitk.GetArrayFromImage(sitkLabel)
            label_array = np.uint8(label_array)
            resampled_label_array = resample_np_array(
                label_array, old_spacing=orig_spacing, interp=sitk.sitkNearestNeighbor, keep_z_spacing=True, new_spacing=new_spacing)

            # change RV labels from 1 to 3 and LV from 3 to 1
            resampled_label_array = (resampled_label_array == 3) * 1 + \
                (resampled_label_array == 2) * 2 + (resampled_label_array == 1) * 3

            # save images as nrrd
            img_file_path = join(output_dir, '{}_img.nrrd'.format(pid))
            seg_file_path = join(output_dir, '{}_seg.nrrd'.format(pid))

            image = sitk.GetImageFromArray(resampled_image_array)
            image.SetSpacing((new_spacing[0], new_spacing[1], orig_spacing[2]))
            sitk.WriteImage(image, img_file_path)

            seg = sitk.GetImageFromArray(resampled_label_array)
            seg.SetSpacing((new_spacing[0], new_spacing[1], orig_spacing[2]))
            sitk.WriteImage(seg, seg_file_path)

            # all_images_list.append(cropped_image)
    print('average time:', np.round(total_bias_correct_time / count, 3))
