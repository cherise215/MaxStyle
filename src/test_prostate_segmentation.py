'''
train a model on various test datasets
'''
import os
from os.path import join
import pandas as pd
from scipy import stats
import numpy as np
import sys
sys.path.append('.')
from src.models.advanced_triplet_recon_segmentation_model import AdvancedTripletReconSegmentationModel
from src.test_basic_segmentation_solver import TestSegmentationNetwork

from src.dataset_loader.prostate_Decathlon_dataset import ProstateDataset
from src.dataset_loader.cardiac_general_dataset import Cardiac_General_Dataset
from src.dataset_loader.transform import Transformations
from src.common_utils.basic_operations import check_dir


pad_size = [288, 288, 1]
crop_size = [224, 224, 1]
new_spacing = [0.625, 0.625, 3.6]


def get_testset(test_dataset_name,test_root_dir= "/vol/biomedic3/cc215/data/prostate_multi_domain_data/reorganized",new_spacing=None,compute_on_clipped=True,intensity_norm_type='min_max'):
    data_aug_policy_name = 'no_aug'
    tr = Transformations(data_aug_policy_name=data_aug_policy_name, pad_size=pad_size,
                         crop_size=crop_size).get_transformation()
    IDX2CLASS_DICT = {
        0: 'BG',
        1: 'FG',

    }
    formalized_label_dict = IDX2CLASS_DICT
    testset_list = []
    normalize_3D = False  # disable it for preprocessed data
    if test_dataset_name in ['G-MedicalDecathlon', 'A-ISBI', 'B-ISBI_1.5', 'C-I2CVB', 'D-UCL', 'E-BIDMC', 'F-HK']:
        IMAGE_FORMAT_NAME = '{pid}/t2_img_clipped.nii.gz'
        LABEL_FORMAT_NAME = '{pid}/label_clipped.nii.gz'
  
        root_folder = f'{test_root_dir}/{test_dataset_name}'

        if test_dataset_name == 'G-MedicalDecathlon':
            dataset = ProstateDataset(transform=tr['validate'],
                                      root_dir=root_folder,
                                      num_classes=2,
                                      idx2cls_dict={0: 'BG', 1: 'FG'},
                                      use_cache=True,
                                      data_setting_name='all',
                                      split='test',
                                      cval=0,
                                      keep_orig_image_label_pair=False,
                                      image_format_name=IMAGE_FORMAT_NAME,
                                      label_format_name=LABEL_FORMAT_NAME, new_spacing=new_spacing, normalize=normalize_3D,intensity_norm_type=intensity_norm_type)
        else:
            root_folder = f'{test_root_dir}/{test_dataset_name}'
            dataset = Cardiac_General_Dataset(root_dir=root_folder,
                                         dataset_name=test_dataset_name,
                                         transform=tr['validate'], num_classes=2, formalized_label_dict=formalized_label_dict,
                                         idx2cls_dict=IDX2CLASS_DICT,
                                         image_format_name=IMAGE_FORMAT_NAME,
                                         label_format_name=LABEL_FORMAT_NAME,
                                         new_spacing=new_spacing, normalize=normalize_3D,intensity_norm_type=intensity_norm_type) # disable it when data has been preprocessed

    else:
        raise NotImplementedError

    return dataset


def evaluate(method_name, segmentation_model, maximum_batch_size, test_dataset_name, test_root_dir,normalize_2D=False,intensity_norm_type='min_max',crop_size = [192, 192, 1],test_set_ratio=1.0, metrics_list=['Dice'],
             save_report_dir=None,
             save_predict=False, save_soft_prediction=False, foreground_only=True,new_spacing=None,compute_on_clipped=True,n_iter=1):
    # evaluation settings
    save_path = save_report_dir + f'/{test_dataset_name}_{str(test_set_ratio)}'
    check_dir(save_path, create=True)
 
    summary_report_file_name = 'iter_{}_summary.csv'.format(n_iter)
    detailed_report_file_name = 'iter_{}_detailed.csv'.format(n_iter)

    test_dataset = get_testset(test_dataset_name,test_root_dir,compute_on_clipped=True,new_spacing=new_spacing,intensity_norm_type=intensity_norm_type)
    tester = TestSegmentationNetwork(test_dataset=test_dataset,
                                     crop_size=crop_size,
                                     maximum_batch_size=maximum_batch_size, segmentation_model=segmentation_model, use_gpu=True,
                                     save_path=save_path, summary_report_file_name=summary_report_file_name, sample_ratio_for_testing=test_set_ratio,
                                     detailed_report_file_name=detailed_report_file_name, patient_wise=True, metrics_list=metrics_list,
                                     foreground_only=foreground_only, normalize_2D=normalize_2D,new_spacing=new_spacing,
                                     save_prediction=save_predict, save_soft_prediction=save_soft_prediction)


    tester.run()
    print('<Summary> {} on dataset {}'.format(method_name, test_dataset_name))
    print(tester.df.describe())
    means = [round(v, 4) for k, v in tester.df.mean(axis=0).items()]
    stds = [round(v, 4) for k, v in tester.df.std(axis=0).items()]
    print('means:', means)
    print('stds:', stds)
    return means, stds, tester.df


