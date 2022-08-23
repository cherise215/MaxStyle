import numpy as np
import os
from os.path import join
import glob
import matplotlib
import matplotlib.pyplot as plt
import torch
import pandas as pd
from random import randint
import SimpleITK as sitk
from torchio.transforms import RandomMotion, RandomSpike, RandomGhosting, RandomBiasField

sys.path.append('../')
from src.common_utils.basic_operations import check_dir, rescale_intensity, crop_or_pad, load_img_label_from_path, recover_image
from src.common_utils.save import save_medical_image


def preprocess3D(image, min_val=0, max_val=1):
    global device
    '''
    preprocess 3D data
    :param image: 3D array
    :param label: 3D array
    '''
    output = np.zeros_like(image, dtype=image.dtype)
    for idx in range(image.shape[0]):  #
        slice_data = image[idx]
        a_min_val, a_max_val = np.percentile(slice_data, (0, 100))
        # restrict the intensity range
        slice_data[slice_data <= a_min_val] = a_min_val
        slice_data[slice_data >= a_max_val] = a_max_val
        # perform normalisation
        scale = (max_val - min_val) / (a_max_val - a_min_val)
        bias = max_val - scale * a_max_val
        output[idx] = slice_data * scale + bias
    return output


if __name__ == '__main__':
    # hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_internal_output = True

    # set up test image data path
    dataset_root = '/vol/biomedic3/cc215/Project/DeformADA/Data/bias_corrected_and_normalized'
    frame = 'ES'
    sub_dir = join(dataset_root, frame)

    id_list = [7, 8, 9, 10, 27, 28, 29, 30, 47,
               48, 49, 50, 67, 68, 69, 70, 87, 88, 89, 90]
    test_image_paths = [[str(pid).zfill(3), join(
        sub_dir, '{}_img.nrrd').format(str(pid).zfill(3))] for pid in id_list]
    test_label_paths = [[str(pid).zfill(3), join(
        sub_dir, '{}_seg.nrrd').format(str(pid).zfill(3))] for pid in id_list]
    fix_img_size = [192, 192]

    attackers = {
        'RandomMotion': RandomMotion(degrees=30, translation=10),
        'RandomSpike': RandomSpike(),
        'RandomGhosting': RandomGhosting(),
        'RandomBias': RandomBiasField()
    }
    n_augmented = 3
    # output
    save_dir = '/vol/biomedic3/cc215/Project/DeformADA/Data/ACDC_artefacted'
    check_dir(save_dir, create=True)
    for attack_name, attacker in attackers.items():
        for j in range(n_augmented):
            for i in range(len(test_image_paths)):
                image, sitkImage = load_img_label_from_path(
                    img_path=test_image_paths[i][1])
                image, label, h_s, w_s, original_h, original_w = crop_or_pad(
                    image=image, label=None, crop_size=fix_img_size)
                image = preprocess3D(image)
                origin_image3D_tensor = torch.tensor(
                    image[:, np.newaxis, :, :], requires_grad=False).float()

                attacked_image = attacker(
                    origin_image3D_tensor.permute(1, 0, 2, 3))  # NCHW->CNHW
                attacked_image = attacked_image.permute(
                    1, 0, 2, 3)  # CNHW->NCHW

                attacked_image = rescale_intensity(
                    attacked_image, new_min=0, new_max=1)
                attacked_image = attacked_image.to(device)
                pid = test_label_paths[i][0]

                print('pid: {} n:{}'.format(pid, j))

                # save 3D images to the dir:
                patient_dir = join(
                    save_dir, *[attack_name, str(pid) + '_' + str(j)])
                check_dir(patient_dir, create=True)
                print('save to', patient_dir)
                # recover size:
                file_path = join(patient_dir, '{}_img.nrrd'.format(frame))
                print(file_path)
                adv_image = recover_image(attacked_image.data.cpu().numpy()[
                    :, 0, :, :], h_s, w_s, original_h, original_w)

                save_medical_image(sitk.GetImageFromArray(
                    adv_image), output_path=file_path, refImage=sitkImage)

                # label path
                label_path = test_label_paths[i][1]
                target_label_path = join(
                    patient_dir, '{}_label.nrrd'.format(frame))

                if os.path.islink(target_label_path):
                    os.unlink(target_label_path)
                os.symlink(label_path, target_label_path)
