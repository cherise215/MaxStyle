'''
train a model on various test datasets
'''
import os
from os.path import join
import pandas as pd
from scipy import stats
import numpy as np
import sys

# sys.path.append('/vol/biomedic3/cc215/Project/MedSeg/')
sys.path.append('.')

from src.models.advanced_triplet_recon_segmentation_model import AdvancedTripletReconSegmentationModel
from src.test_basic_segmentation_solver import TestSegmentationNetwork

from src.dataset_loader.cardiac_ACDC_dataset import CardiacACDCDataset
from src.dataset_loader.cardiac_general_dataset import Cardiac_General_Dataset
from src.dataset_loader.transform import Transformations
from src.dataset_loader.base_segmentation_dataset import ConcatDataSet
from src.common_utils.basic_operations import check_dir


pad_size = [224, 224, 1]
crop_size = [192, 192, 1]
# new_spacing = [1.36719, 1.36719, -1]


def get_testset(test_dataset_name, test_root_dir="",frames=['ED', 'ES'],new_spacing = [1.36719, 1.36719, -1],intensity_norm_type='min_max'):
    data_aug_policy_name = 'no_aug'
    tr = Transformations(data_aug_policy_name=data_aug_policy_name, pad_size=pad_size,
                         crop_size=crop_size).get_transformation()
    IDX2CLASS_DICT = {
        0: 'BG',
        1: 'LV',
        2: 'MYO',
        3: 'RV',
    }
    formalized_label_dict = IDX2CLASS_DICT
    testset_list = []
    if test_dataset_name in ['ACDC', 'MM', 'RandomGhosting', 'RandomBias', 'RandomSpike', 'RandomMotion']:
        # dataset with two frames
        for frame in frames:
            if test_dataset_name == 'ACDC':
                root_dir = '/vol/biomedic3/cc215/Project/DeformADA/Data/bias_corrected_and_normalized'
                test_dataset = CardiacACDCDataset(root_dir=root_dir, transform=tr['validate'], idx2cls_dict=IDX2CLASS_DICT, num_classes=4,
                                                  data_setting_name='10', formalized_label_dict=formalized_label_dict,
                                                  subset_name=frame, split='test', myocardium_seg=False,
                                                  right_ventricle_seg=False,
                                                  new_spacing=None, normalize=False,intensity_norm_type=intensity_norm_type)
            elif test_dataset_name == 'MM':
                root_dir = '/vol/biomedic3/cc215/data/MICCAI2021_multi_domain_robustness_datasets/MM'
                IMAGE_FORMAT_NAME = '{pid}/' + frame + '_img.nii.gz'
                LABEL_FORMAT_NAME = '{pid}/' + frame + '_seg.nii.gz'
                test_dataset = Cardiac_General_Dataset(root_dir=root_dir,
                                                  transform=tr['validate'], num_classes=4, formalized_label_dict=formalized_label_dict,
                                                  idx2cls_dict=IDX2CLASS_DICT,
                                                  image_format_name=IMAGE_FORMAT_NAME,
                                                  label_format_name=LABEL_FORMAT_NAME,
                                                   new_spacing=None, normalize=False,intensity_norm_type=intensity_norm_type) # disable it when data has been preprocessed

            elif test_dataset_name in ['RandomGhosting', 'RandomBias', 'RandomSpike', 'RandomMotion']:
                root_folder = '/vol/biomedic3/cc215/data/ACDC/ACDC_artefacted/{}'.format(test_dataset_name)
                IMAGE_FORMAT_NAME = '{pid}/' + frame + '_img.nrrd'
                LABEL_FORMAT_NAME = '{pid}/' + frame + '_label.nrrd'
                test_dataset = Cardiac_General_Dataset(root_dir=root_folder,
                                                  dataset_name=test_dataset_name,
                                                  transform=tr['validate'], num_classes=4, formalized_label_dict=formalized_label_dict,
                                                  idx2cls_dict=IDX2CLASS_DICT,
                                                  image_format_name=IMAGE_FORMAT_NAME,
                                                  label_format_name=LABEL_FORMAT_NAME,
                                                  new_spacing=None, normalize=False,intensity_norm_type=intensity_norm_type)  # disable it when data has been preprocessed

            else:
                raise NotImplementedError
            testset_list.append(test_dataset)
    elif 'MSCMRSeg' in test_dataset_name:
        root_folder = '/vol/biomedic3/cc215/data/MSCMRSeg_resampled/'
        if test_dataset_name == 'MSCMRSeg_C0':
            IMAGE_FORMAT_NAME = '{pid}/C0/image_corrected.nii.gz'
            LABEL_FORMAT_NAME = '{pid}/C0/label_corrected.nii.gz'
        elif test_dataset_name == 'MSCMRSeg_LGE':
            IMAGE_FORMAT_NAME = '{pid}/LGE/image_corrected.nii.gz'
            LABEL_FORMAT_NAME = '{pid}/LGE/label_corrected.nii.gz'
        elif test_dataset_name == 'MSCMRSeg_T2':
            IMAGE_FORMAT_NAME = '{pid}/T2/image_corrected.nii.gz'
            LABEL_FORMAT_NAME = '{pid}/T2/label_corrected.nii.gz'
        else:
            raise NotImplementedError
        test_dataset = Cardiac_General_Dataset(root_dir=root_folder,
                                          dataset_name=test_dataset_name,
                                          transform=tr['validate'], num_classes=4, formalized_label_dict=formalized_label_dict,
                                          idx2cls_dict=IDX2CLASS_DICT,
                                          image_format_name=IMAGE_FORMAT_NAME,
                                          label_format_name=LABEL_FORMAT_NAME,
                                           new_spacing=None, normalize=False,intensity_norm_type=intensity_norm_type)  # disable it when data has been preprocessed

        testset_list.append(test_dataset)
    elif 'MnM-2' == test_dataset_name:
        root_folder = '/vol/biomedic3/cc215/data/MnM-2/preprocessed/training'
        for frame in ['ED', 'ES']:
            IMAGE_FORMAT_NAME = '{pid}/SA_' + frame + '.nii.gz'
            LABEL_FORMAT_NAME = '{pid}/SA_' + frame + '_gt.nii.gz'
            test_dataset = Cardiac_General_Dataset(root_dir=root_folder,
                                              dataset_name=test_dataset_name,
                                              transform=tr['validate'], num_classes=4, formalized_label_dict=formalized_label_dict,
                                              idx2cls_dict=IDX2CLASS_DICT,
                                              image_format_name=IMAGE_FORMAT_NAME,
                                              label_format_name=LABEL_FORMAT_NAME,
                                               new_spacing=None, normalize=False,intensity_norm_type=intensity_norm_type)  # disable it when data has been preprocessed to save time

            testset_list.append(test_dataset)
    elif 'UKBB' == test_dataset_name:
        root_folder = '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test/'
        for frame in ['ED', 'ES']:
            IMAGE_FORMAT_NAME = '{pid}'+f'/sa_{frame}.nii.gz'
            LABEL_FORMAT_NAME = '{pid}'+f'/label_sa_{frame}.nii.gz'
            test_dataset = Cardiac_General_Dataset(root_dir=root_folder,
                                              dataset_name=test_dataset_name,
                                              transform=tr['validate'], num_classes=4, formalized_label_dict=formalized_label_dict,
                                              idx2cls_dict=IDX2CLASS_DICT,
                                              image_format_name=IMAGE_FORMAT_NAME,
                                              label_format_name=LABEL_FORMAT_NAME,
                                              new_spacing=None, normalize=False,intensity_norm_type=intensity_norm_type)  # disable it when data has been preprocessed to save time

            testset_list.append(test_dataset)
    else:
        raise NotImplementedError

    if len(testset_list) >= 2:
        concatdataset = ConcatDataSet(testset_list)
    else:
        concatdataset = testset_list[0]
    return concatdataset


def evaluate(method_name, segmentation_model, maximum_batch_size, test_dataset_name, test_root_dir,
             crop_size=[192,192,1],normalize_2D=False,new_spacing=None, test_set_ratio=1.0, frames=['ED', 'ES'],
              metrics_list=['Dice'],
             save_report_dir=None,
             save_predict=False, save_soft_prediction=False, foreground_only=False,intensity_norm_type='min_max'):
    n_iter = segmentation_model.n_iter
    # evaluation settings
    save_path = save_report_dir + f'/{test_dataset_name}_{str(test_set_ratio)}'
    check_dir(save_path, create=True)
    if 'TTA' in method_name:
        summary_report_file_name = '{}_iter_{}_summary.csv'.format(method_name, n_iter)
        detailed_report_file_name = '{}_iter_{}_detailed.csv'.format(method_name, n_iter)
    else:
        summary_report_file_name = 'iter_{}_summary.csv'.format(n_iter)
        detailed_report_file_name = 'iter_{}_detailed.csv'.format(n_iter)

    test_dataset = get_testset(test_dataset_name,test_root_dir=test_root_dir, frames=frames,new_spacing=new_spacing,intensity_norm_type=intensity_norm_type)
    tester = TestSegmentationNetwork(test_dataset=test_dataset,
                                     crop_size=crop_size,
                                     maximum_batch_size=maximum_batch_size, segmentation_model=segmentation_model, use_gpu=True,
                                     save_path=save_path, summary_report_file_name=summary_report_file_name, sample_ratio_for_testing=test_set_ratio,
                                     detailed_report_file_name=detailed_report_file_name, patient_wise=True, metrics_list=metrics_list,
                                     foreground_only=foreground_only, normalize_2D=normalize_2D,new_spacing=new_spacing,
                                     save_prediction=save_predict, save_soft_prediction=save_soft_prediction)

    tester.run()

    print('<Summary> {} on dataset {} across {}'.format(method_name, test_dataset_name, str(frames)))
    print(tester.df.describe())
    # # save each method's result summary/details on each test dataset
    # tester.df.describe().to_csv(join(save_path + f'/{test_dataset_name}' + '{}_{}_iter_{}_summary.csv'.format(
    #     method_name, str(frames), str(n_iter))))
    # tester.df.to_csv(join(save_path + f'/{test_dataset_name}' + '{}_{}_iter_{}_detailed.csv'.format(
    #     method_name, str(frames), str(n_iter))))

    means = [round(v, 4) for k, v in tester.df.mean(axis=0).items()]
    stds = [round(v, 4) for k, v in tester.df.std(axis=0).items()]
  
    n_classes = segmentation_model.num_classes
    if len(means) == n_classes:
        means = means[1:]
    if len(stds) ==n_classes:
        stds = stds[1:]
    print('means:', means)
    print('stds:', stds)
    return means, stds, tester.df


