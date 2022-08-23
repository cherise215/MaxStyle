# Created by cc215 at 27/1/20
# this dataset is available at '/vol/medic01/users/cc215/data/MedicalDecathlon/Task05_Prostate/preprocessed'
# from Medical Decathlon challenge dataset, we use T2 as input for the task.
# Note: All images have been preprocessed (resampled to the 0.625 x 0.625 x 3.6 mm, the median value of the voxel spacings), following the preprocessing steps in
# "nn-Unet"
# contains 32 patients in total
# Data structure:
# each patient has a nrrd file
# path:
# image : root_dir/ES/{patient_id}/t2_img.nrrd
# label : root_dir/ES/{patient_id}/label.nrrd

import logging
import numpy as np
import os
import SimpleITK as sitk
import torch
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../')
from src.dataset_loader.base_segmentation_dataset import BaseSegDataset
from src.common_utils.basic_operations import switch_kv_in_dict,intensity_norm_fn_selector
from src.common_utils.data_structure import Cache
from src.common_utils.basic_operations import load_img_label_from_path, crop_or_pad,check_dir, rescale_intensity,load_dict,save_dict

DATASET_NAME = 'Prostate'
IDX2CLASS_DICT = {
    0: 'BG',
    1: 'PZ',
    2: 'CZ',
}
IMAGE_FORMAT_NAME = '{pid}/t2_img_clipped.nii.gz'
LABEL_FORMAT_NAME = '{pid}/label_clipped.nii.gz'
IMAGE_SIZE = (320, 320, 1)
LABEL_SIZE = (320, 320)


class ProstateDataset(BaseSegDataset):
    def __init__(self,
                 transform, dataset_name=DATASET_NAME,
                 root_dir='/vol/biomedic3/cc215/data/prostate_multi_domain_data/reorganized/G-MedicalDecathlon',
                 num_classes=2,
                 image_size=IMAGE_SIZE,
                 label_size=LABEL_SIZE,
                 idx2cls_dict=IDX2CLASS_DICT,
                 use_cache=True,
                 data_setting_name='all',
                 split='train',
                 cval=0,  # int, seed for controlling the train/validation split
                 formalized_label_dict=None,
                 keep_orig_image_label_pair=True,
                 image_format_name=IMAGE_FORMAT_NAME,
                 label_format_name=LABEL_FORMAT_NAME,
                 binary_segmentation=False,
                 smooth_label=False,ignore_black_slice=True,
                 debug=False, new_spacing=None, crop_size=[224,224,1],normalize=True,intensity_norm_type="min_max"
                 ):
        # predefined variables
        # initialization
        self.debug = debug
        self.data_setting_name = data_setting_name
        self.split = split  # can be validation or test or all
        self.cval = cval
        super(ProstateDataset, self).__init__(
            root_dir=root_dir,image_format_name=image_format_name,label_format_name=label_format_name,binary_segmentation=binary_segmentation,
            dataset_name=dataset_name, transform=transform, num_classes=num_classes,ignore_black_slice=ignore_black_slice,
                                              image_size=image_size, label_size=label_size, idx2cls_dict=idx2cls_dict,intensity_norm_type=intensity_norm_type,
                                              use_cache=use_cache, crop_size=crop_size,smooth_label=  smooth_label,new_spacing=new_spacing,normalize=normalize,formalized_label_dict=formalized_label_dict, keep_orig_image_label_pair=keep_orig_image_label_pair)
        self.datasize, self.patient_id_list, self.index2pid_dict, self.index2slice_dict,self.pid2spacing_dict =self.scan_dataset(use_cache=False)
        self.temp_data_dict = None  # temporary data during loading
        self.index = 0  # index for selecting which slices
        self.pid = self.patient_id_list[0]  # current pid
        self.patient_number = len(self.patient_id_list)
        self.slice_id = self.index2slice_dict[0]

        self.dataset_name = DATASET_NAME + '_{}_{}'.format(str(data_setting_name), split)
        if self.split == 'train':
            self.dataset_name += str(cval)

        print('load {},  containing {}, found {} slices'.format(
            self.root_dir, len(self.patient_id_list), self.datasize))
        

    def get_voxel_spacing(self):
        '''
        return the current id
        :return:
        '''
        if self.new_spacing is not None:
            return self.new_spacing
        else:
            if self.pid2spacing_dict is None:
                return  self.pid2spacing_dict[self.patient_id_list[self.index]]
            else:
                print ('default voxel spacing')
                return [0.625, 0.625, 3.6]
  
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
            index2spacing_dict = cache_dict['index2spacing']
        else:
            patient_id_list = self.get_pid_list(identifier=self.data_setting_name, cval=self.cval)[self.split]

            # print ('{} set has {} patients'.format(self.split,len(patient_id_list)))
            index2pid_dict = {}
            index2slice_dict = {}
            index2spacing_dict = {}
            cur_ind = 0
            for pid in patient_id_list:
                img_path = os.path.join(self.root_dir, self.image_format_name.format(pid=pid))
                label_path = os.path.join(self.root_dir, self.label_format_name.format(pid=pid))
                try:
                    ndarray, label_arr, sitkImage, sitkLabel = load_img_label_from_path(img_path, label_path, None, normalize=False)                
                    if self.new_spacing is not None:
                        spacing = list(self.new_spacing)
                    else:
                        spacing = list(sitkImage.GetSpacing())
                    num_slices = ndarray.shape[0]
                    if num_slices!=label_arr.shape[0]:
                        print ('image and label slice number not match, found {} slices in image, {} slices in label'.format(num_slices,label_arr.shape[0]))
                        continue
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
                            index2spacing_dict[cur_ind]=spacing
                            cur_ind += 1
                except IOError:
                    print(f'error in loading image and label for pid:{pid},{img_path}')
            datasize = cur_ind

            cache_dict = {
                        'datasize': datasize,
                        'patient_id_list': patient_id_list,
                        'index2slice_dict': index2slice_dict,
                        'index2patientid': index2pid_dict,
                        'index2spacing':index2spacing_dict 
                    }
            save_dict(cache_dict, file_path=cache_file_path) 
        # self.cache_dict  =cache_dict        
        return datasize, patient_id_list, index2pid_dict, index2slice_dict,index2spacing_dict
      

    def get_pid_list(self, identifier, cval):
        assert cval >= 0, 'cval must be >0'
        all_p_id_list = sorted(os.listdir(self.root_dir))
        test_ids = ['patient_17', 'patient_7', 'patient_12', 'patient_22', 'patient_0',
                    'patient_24', 'patient_5']
        train_val_ids = list(set(all_p_id_list) - set(test_ids))
        train_ids, val_ids = train_test_split(train_val_ids, test_size=0.1, random_state=cval)
        size = len(train_val_ids)
        labelled_ids = train_ids[:(size // 2)]
        unlabelled_ids = train_ids[(size // 2):]
        if identifier == 'all':
            # use all training data as labelled data
            labelled_ids_split = train_ids
            unlabelled_ids = []
        elif identifier == 'three_shot':
            labelled_ids_split, _ = train_test_split(labelled_ids, train_size=3, random_state=cval)
        elif identifier == 'three_shot_upperbound':
            labelled_ids_split, _ = train_test_split(labelled_ids, train_size=3, random_state=cval)
            labelled_ids_split = labelled_ids_split + unlabelled_ids
            unlabelled_ids = []
        elif identifier == 'full':
            labelled_ids_split = labelled_ids
        elif isinstance(float(identifier), float):
            identifier = float(identifier)
            if 0 < identifier < 1:
                labelled_ids_split, _ = train_test_split(labelled_ids, train_size=identifier, random_state=cval)
            elif identifier > 1:
                identifier = int(identifier)
                if 0 < identifier < len(labelled_ids):
                    labelled_ids_split, _ = train_test_split(labelled_ids, train_size=identifier, random_state=cval)
                elif abs(identifier + 1) < 1e-6:
                    labelled_ids_split = labelled_ids
                else:
                    raise ValueError
            else:
                raise NotImplementedError
        else:
            print('use all training subjects')
            labelled_ids_split = labelled_ids
        return {
            'name': str(identifier) + '_cv_' + str(cval),
            'train': labelled_ids_split,
            'validate': val_ids,
            'test': test_ids,
            'test+unlabelled': test_ids + unlabelled_ids,
            'unlabelled': unlabelled_ids,
        }

    


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.dataset_loader.transform import Transformations  #
    from torch.utils.data import DataLoader
    import numpy as np
    np.random.seed(1234)
    image_size = [320, 320, 1]
    label_size = [320, 320]
    crop_size = [224, 224, 1]
    tr = Transformations(data_aug_policy_name='Prostate_affine_elastic_intensity',
                         pad_size=image_size, crop_size=crop_size).get_transformation()
    dataset = ProstateDataset(root_dir='/vol/biomedic3/cc215/data/prostate_multi_domain_data/reorganized/G-MedicalDecathlon', 
                                split='train', data_setting_name=0.3, transform=tr['train'], binary_segmentation=True,new_spacing=None,use_cache=False,
                              num_classes=3,ignore_black_slice=True)
    train_loader = DataLoader(dataset=dataset, num_workers=0, batch_size=1, shuffle=False, drop_last=True)
    print(len(train_loader))
    for i, item in enumerate(train_loader):
        print ('orginal image shape here')

        img = item['origin_image']
        label = item['origin_label']
        print(i, dataset.get_id())
        print(img.numpy().shape)
        print(label.numpy().shape)
        print ('after image shape here')

        plt.subplot(141)
        plt.imshow(img.numpy()[0,0], cmap='gray')
        plt.subplot(142)
        plt.imshow(label.numpy()[0])
        plt.colorbar()

        img = item['image']
        label = item['label']
        print(img.numpy().shape)
        print(label.numpy().shape)
        plt.subplot(143)
        plt.imshow(img.numpy()[0,0], cmap='gray')
        plt.subplot(144)
        plt.imshow(label.numpy()[0])
        plt.colorbar()
        plt.savefig(f"/vol/biomedic3/cc215/Project/MedSeg/log/test/prostate/{i}_new.png")
        if i >= 10:
            break
