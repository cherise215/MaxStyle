# Created by cc215 at 27/1/20
# this dataset is available at '/vol/medic01/users/cc215/Dropbox/projects/DeformADA/Data/ACDC'
# from ACDC challenge dataset, containing ED/ES whole stack (3D).
# Note: All images have been preprocessed and cropped to 224 by 224, following the preprocessing steps in
# "Semi-Supervised and Task-Driven Data Augmentation"
# https://arxiv.org/abs/1902.05396
# contains 100 patients in total
# Data structure:
# each patient has a nrrd file for each phase ED/ES.
# path:
# ED images : root_dir/ED/{patient_id}_img.nrrd,root_dir/ED/{patient_id}_seg.nrrd, all [n_slices*224*224]s
# ES images : root_dir/ES/{patient_id}_img.nrrd,root_dir/ES/{patient_id}_seg.nrrd, all [n_slices*224*224]s

import numpy as np
import os
import random
import SimpleITK as sitk
import sys

sys.path.append('../')
from src.common_utils.basic_operations import load_img_label_from_path, crop_or_pad,check_dir, rescale_intensity,load_dict,save_dict
from src.dataset_loader.base_segmentation_dataset import BaseSegDataset
from src.dataset_loader.ACDC_few_shot_cv_settings import get_ACDC_split_policy
from src.common_utils.basic_operations import load_img_label_from_path, crop_or_pad, rescale_intensity
DATASET_NAME = 'ACDC'
IDX2CLASS_DICT = {
    0: 'BG',
    1: 'LV',
    2: 'MYO',
    3: 'RV',
}
IMAGE_FORMAT_NAME = '{pid}_img.nrrd'
LABEL_FORMAT_NAME = '{pid}_seg.nrrd'
IMAGE_SIZE = (224, 224, 1)
LABEL_SIZE = (224, 224)

# images are stored like
# image: root_dir/subset_name/{p_id}_img.nrrd
# label: root_dir/subset_name/{p_id}_seg.nrrd


class CardiacACDCDataset(BaseSegDataset):
    def __init__(self,
                 transform, dataset_name=DATASET_NAME,
                 root_dir='/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized',
                 subset_name='ES', num_classes=4,
                 image_size=IMAGE_SIZE,
                 label_size=LABEL_SIZE,
                 idx2cls_dict=IDX2CLASS_DICT,
                 use_cache=False,
                 data_setting_name='three_shot',
                 split='train',
                 cval=0,  # int, cross validation id
                 formalized_label_dict=None,
                 binary_segmentation=False,
                 smooth_label=False,
                 myocardium_seg=False,
                 right_ventricle_seg=False,
                 ignore_black_slice=True,
                 keep_orig_image_label_pair=True,
                 image_format_name=IMAGE_FORMAT_NAME,
                 label_format_name=LABEL_FORMAT_NAME,
                 crop_size=[192,192,1],
                 new_spacing=[1.36719, 1.36719, -1], 
                 intensity_norm_type = "min_max",
                 normalize=True,
                 debug=False,

                 ):
        # predefined variables
        # initialization

        self.debug = debug
        self.data_setting_name = data_setting_name
        self.split = split  # can be validation or test or all
        self.cval = cval
        self.myocardium_seg = myocardium_seg
        self.right_ventricle_seg = right_ventricle_seg
        if myocardium_seg:
            formalized_label_dict = {0: 'BG', 1: 'MYO'}
        if right_ventricle_seg:
            formalized_label_dict = {0: 'BG', 1: 'RV'}

        self.myocardium_seg
        root_dir = os.path.join(root_dir, subset_name)
        self.subset_name = subset_name
        super(CardiacACDCDataset, self).__init__(root_dir=root_dir,image_format_name=image_format_name,label_format_name=label_format_name,dataset_name=dataset_name, transform=transform, num_classes=num_classes,
                                              image_size=image_size, label_size=label_size, idx2cls_dict=idx2cls_dict,
                                              use_cache=use_cache, formalized_label_dict=formalized_label_dict,
                                            keep_orig_image_label_pair=keep_orig_image_label_pair,
                                            intensity_norm_type=intensity_norm_type, normalize=normalize,new_spacing=new_spacing,crop_size=crop_size,
                                            binary_segmentation=binary_segmentation,smooth_label=smooth_label, ignore_black_slice=ignore_black_slice)
                                                        
        self.datasize, self.patient_id_list, self.index2pid_dict, self.index2slice_dict,self.pid2spacing_dict =self.scan_dataset(use_cache=False)

        self.temp_data_dict = None  # temporary data during loading
        self.index = 0  # index for selecting which slices
        self.pid = self.patient_id_list[0]  # current pid
        self.patient_number = len(self.patient_id_list)
        self.slice_id = self.index2slice_dict[0]

        self.dataset_name = DATASET_NAME + '_{}_{}_{}'.format(subset_name, data_setting_name, split)
        if self.split == 'train':
            self.dataset_name += str(cval)

        print('load {},  containing {}, found {} slices'.format(
            self.dataset_name + self.subset_name, len(self.patient_id_list), self.datasize))
    
    def scan_dataset(self,use_cache=False):
        '''
        given the data setting names and split, cross validation id
        :return: dataset size, a list of pids for training/testing/validation,
         and a dict for retrieving patient id and slice id.
        '''
        cache_dir = './log/cache/'
        check_dir(cache_dir, create=True)
        cache_file_name = self.root_dir.replace('/', '_')+self.image_format_name.replace('/', '_')+self.label_format_name.replace('/', '_')+str(self.data_setting_name)+str(self.cval)+self.split+'.pkl'
        cache_file_path = os.path.join(cache_dir, cache_file_name)
        self.cache_file_path=cache_file_path
        if use_cache and os.path.exists(cache_file_path):
            print('load basic information from cache:', cache_file_path)
            cache_dict = load_dict(cache_file_path)
            datasize = cache_dict['datasize']
            patient_id_list = cache_dict['patient_id_list']
            index2slice_dict = cache_dict['index2slice_dict']
            index2pid_dict = cache_dict['index2patientid']
            pid2spacing_dict = cache_dict['pid2spacing']
        else:
            
            patient_id_list = get_ACDC_split_policy(identifier=self.data_setting_name, cval=self.cval)[self.split]
            # print ('{} set has {} patients'.format(self.split,len(patient_id_list)))
            index2pid_dict = {}
            index2slice_dict = {}
            pid2spacing_dict = {}
            cur_ind = 0
            for pid in patient_id_list:
                img_path = os.path.join(self.root_dir, self.image_format_name.format(pid=pid))
                label_path = os.path.join(self.root_dir, self.label_format_name.format(pid=pid))
                try:
                    ndarray, label_arr, sitkImage, sitkLabel = load_img_label_from_path(img_path, label_path, self.new_spacing, normalize=False)                
                    if self.new_spacing is not None:
                        spacing = self.new_spacing
                    else:
                        spacing = sitkImage.GetSpacing()
                    num_slices = ndarray.shape[0]
                    if num_slices!=label_arr.shape[0]:
                        print ('image and label slice number not match, found {} slices in image, {} slices in label'.format(num_slices,label_arr.shape[0]))
                        continue
                    pid2spacing_dict[pid]=spacing

                    for cnt in range(num_slices):
                        if self.ignore_black_slice:
                            img_slice_data = ndarray[cnt, :, :]
                            img_slice_data -= np.mean(img_slice_data)
                            if np.sum(abs(img_slice_data) - 0) > 1e-4:
                                index2pid_dict[cur_ind] = pid
                                index2slice_dict[cur_ind] = cnt
                                cur_ind += 1
                        else:
                            index2pid_dict[cur_ind] = pid
                            index2slice_dict[cur_ind] = cnt
                            cur_ind += 1
                except IOError:
                    print(f'error in loading image and label for pid:{pid},{img_path}')
            datasize = cur_ind

            cache_dict = {
                        'datasize': datasize,
                        'patient_id_list': patient_id_list,
                        'index2slice_dict': index2slice_dict,
                        'index2patientid': index2pid_dict,
                        'pid2spacing':pid2spacing_dict 
                    }
            save_dict(cache_dict, file_path=cache_file_path) 
        # self.cache_dict  =cache_dict        
        return datasize, patient_id_list, index2pid_dict, index2slice_dict,pid2spacing_dict

    def get_id(self):
        '''
        return the current patient id
        :return:
        '''
        return self.patient_id_list[self.index] + "_" + self.subset_name
    def get_spacing(self):
        '''
        return the current spacing
        :return:
        '''
        assert self.index2spacing_dict is not None
        return self.index2spacing_dict[self.index]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.dataset_loader.transform import Transformations  #
    from torch.utils.data import DataLoader

    image_size = [224, 224, 1]
    label_size = [224, 224]
    crop_size = [192, 192, 1]
    tr = Transformations(data_aug_policy_name='UKBB_affine_elastic_intensity_aug',
                         pad_size=image_size, crop_size=crop_size).get_transformation()
    dataset = CardiacACDCDataset(data_setting_name='standard', split='train',
                                 transform=tr['train'], num_classes=4, right_ventricle_seg=False,use_cache=False,
                                 intensity_norm_type="min_max",
                                 root_dir='/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized',keep_orig_image_label_pair=True, new_spacing=[1.25,1.25,10])
    train_loader = DataLoader(dataset=dataset, num_workers=0, batch_size=1, shuffle=False, drop_last=True)

    for i, item in enumerate(dataset):
        img = item['origin_image']
        label = item['origin_label']
        print(img.numpy().shape)
        print(label.numpy().shape)
        print ("min:", img.min())
        print ("max:", img.max())

        plt.subplot(141)
        plt.imshow(img.numpy()[0], cmap='gray')
        plt.subplot(142)
        plt.imshow(label.numpy(),cmap ='jet',interpolation='none')

        img = item['image']
        label = item['label']
        print(img.numpy().shape)
        print(label.numpy().shape)
        print ("augmented min:", img.min())
        print ("augmented max:", img.max())
        plt.subplot(143)
        plt.imshow(img.numpy()[0], cmap='gray')
        plt.subplot(144)
        plt.imshow(label.numpy(),cmap ='jet',interpolation='none')
        plt.savefig('/vol/biomedic3/cc215/Project/MedSeg/log/test/NEW_ACDC_' + str(i) + '.png')
        if i >= 10:
            break
