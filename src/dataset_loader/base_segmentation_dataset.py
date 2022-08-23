# Created by cc215 at 11/12/19
# Enter feature description here
# Enter scenario name here
# Enter steps here
import os
import sys
import torch.utils.data as data
import torch
from torch.utils.data import Dataset
import random

import numpy as np
sys.path.append('../')

from src.common_utils.basic_operations import switch_kv_in_dict,intensity_norm_fn_selector
from src.common_utils.data_structure import Cache
from src.common_utils.basic_operations import load_img_label_from_path, crop_or_pad, rescale_intensity


class BaseSegDataset(Dataset):
    def __init__(self, root_dir, image_format_name, label_format_name,dataset_name, transform, image_size, label_size, idx2cls_dict=None, num_classes=2,
                 use_cache=False, formalized_label_dict=None, keep_orig_image_label_pair=False, ignore_black_slice=True,maximum_cache_size=20,intensity_norm_type='min_max',
                 binary_segmentation=False,smooth_label=False,normalize=False,  crop_size=[192,192,1],
                 new_spacing=None):
        '''
        Base 2D dataset class
        :param dataset_name:
        :param transform:
        :param image_size:
        :param label_size:
        :param idx2cls_dict:
        :param num_classes:
        :param use_cache:
        :param formalized_label_dict:
        :param keep_orig_image_label_pair:  if true, then each time will produce image-label pairs before and/after data augmentation
        '''
        super(BaseSegDataset).__init__()
        self.dataset_name = dataset_name
        self.root_dir = root_dir

        self.image_format_name = image_format_name
        self.label_format_name = label_format_name
        self.num_classes = num_classes
        self.image_size = image_size
        self.label_size = label_size
        self.transform = transform
        self.idx2cls_dict = idx2cls_dict
        self.binary_segmentation=binary_segmentation
        self.ignore_black_slice=ignore_black_slice
        self.intensity_norm_fn= intensity_norm_fn_selector(intensity_norm_type)
        self.intensity_norm_type = intensity_norm_type
        self.smooth_label=smooth_label
        self.normalize=normalize
        self.new_spacing = new_spacing
        self.crop_size = crop_size
        if idx2cls_dict is None:
            self.idx2cls_dict = {}
            for i in range(num_classes):
                self.idx2cls_dict[i] = str(i)
        self.formalized_label_dict = self.idx2cls_dict if formalized_label_dict is None else formalized_label_dict
        self.use_cache = use_cache
        self.cache_dict =Cache(maxlen=maximum_cache_size)

        self.index = 0
        self.keep_orig_image_label_pair = keep_orig_image_label_pair
      
        self.datasize, self.patient_id_list, self.index2pid_dict, self.index2slice_dict, self.pid2spacing_dict= self.scan_dataset()
        self.patient_number =len(self.patient_id_list)
        self.temp_data_dict = None  # temporary data during loading
        self.pid = self.patient_id_list[0]  # current pid
        self.patient_number = len(self.patient_id_list)
        self.slice_id = 0
    
    def get_id(self):
        '''
        return the current id
        :return:
        '''
        return self.index

    def get_voxel_spacing(self):
        '''
        return the current id
        :return:
        '''
        if self.new_spacing is not None:
            return self.new_spacing
        else:
            if not self.pid2spacing_dict is None:
                return  self.pid2spacing_dict[self.pid]
            else:
                print ('fake constant voxel spacing')
                return [1,1,1]

    def set_id(self, index):
        '''
        set the current id with semantic information (e.g. patient id)
        :return:
        '''
        return self.index

    def __getitem__(self, index):
        self.set_id(index)
        if self.use_cache:
            # load data from RAM to save IO time
            if index in self.cache_dict.keys():
                data_dict = self.cache_dict[index]
            else:
                old_data_dict = self.load_data(index)
                data_dict = self.preprocess_data_to_tensors(old_data_dict['image'], old_data_dict['label'])
                data_dict["pid"] = old_data_dict["pid"]

                self.cache_dict[index] = data_dict
        else:
            old_data_dict = self.load_data(index)
            data_dict = self.preprocess_data_to_tensors(old_data_dict['image'], old_data_dict['label'])
            data_dict["pid"] = old_data_dict["pid"]
        return data_dict

    def scan_dataset(self,use_cache=False):
        raise NotImplementedError
    def __len__(self):
        return self.datasize
        
    def load_data(self, index):
        '''
        give a index to fetch a data package for one patient
        :return:
        data from a patient.
        class dict: {
        'image': ndarray,H*W*CH, CH =1, for gray images
        'label': ndaray, H*W
        '''
        assert len(self.patient_id_list) > 0, "no data found in the disk at {}".format(self.root_dir)
        patient_id, slice_id = self.find_pid_slice_id(index)
        self.pid = patient_id
        self.slice_id = slice_id
        if self.debug:
            print(patient_id)
        image_3d, label_3d, sitkImage, sitkLabel = self.load_patientImage_from_nrrd(
            patient_id, new_spacing=self.new_spacing, normalize=self.normalize)
        
        max_id = image_3d.shape[0]
        id_list = list(np.arange(max_id))

        image = image_3d[slice_id]
        label = label_3d[slice_id]
        # remove slice w.o objects
        if self.ignore_black_slice:
            while True:
                if abs(np.sum(label) - 0) > 1e-4:
                    break
                else:
                    id_list.remove(slice_id)
                    random.shuffle(id_list)
                    slice_id = id_list[0]
                image = image_3d[slice_id]
                label = label_3d[slice_id]

        if self.smooth_label:
            raise NotImplementedError
        ## add binary segmentation option
        if self.binary_segmentation:
            label[label > 0] = 1
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        if self.debug:
            print(image.shape)
            print(label.shape)
        self.pid = patient_id
        cur_data_dict = {'image': image,
                         'label': label,
                         'pid': patient_id,
                        'new_spacing':self.new_spacing}
        del image_3d,label_3d,sitkImage,sitkLabel
        self.temp_data_dict = cur_data_dict
        return cur_data_dict

    def load_patientImage_from_nrrd(self, patient_id, new_spacing=None, normalize=False):
        if "pid" in self.image_format_name:
            img_name = self.image_format_name.format(pid=patient_id)
            label_name = self.label_format_name.format(pid=patient_id)
            img_path = os.path.join(self.root_dir, img_name)
            label_path = os.path.join(self.root_dir, label_name)
        elif "p_id" in self.image_format_name:
            ## for historical reasons, we use p_id to represent patient id
            img_name = self.image_format_name.format(p_id=patient_id)
            label_name = self.label_format_name.format(p_id=patient_id)
            img_path = os.path.join(self.root_dir, img_name)
            label_path = os.path.join(self.root_dir, label_name)
        else:
            raise ValueError("image_format_name should contain pid or p_id")
        # load data
        img_arr, label_arr, sitkImage, sitkLabel = load_img_label_from_path(
            img_path, label_path, new_spacing=new_spacing, normalize=normalize)

        return img_arr, label_arr, sitkImage, sitkLabel
    
    def preprocess_data_to_tensors(self, image, label):
        '''
        use predefined data preprocessing pipeline to transform data
        :param image: ndarray: H*W*CH
        :param label: ndarray: H*W
        :return:
        dict{
        'image': torch tensor: ch*H*W
        'label': torch tensor: H*W
        }
        '''
        assert len(image.shape) == 3 and len(
            label.shape) <= 3, 'input image and label dim should be 3 and 2 respectively, but got {} and {}'.format(
            len(image.shape),
            len(label.shape))
        # safe check, the channel should be in the last dimension
        assert image.shape[2] < image.shape[1] and image.shape[2] < image.shape[
            0], ' input image should be of the HWC format'
        # reassign label:
        new_labels = self.formulate_labels(label)

        new_labels = np.uint8(new_labels)
        orig_image = image
        orig_label = new_labels.copy()

        # expand label to be 3D for transformation
        if_slice_data = True if len(label.shape) == 2 else False
        if if_slice_data:
            new_labels = new_labels[:, :, np.newaxis]
        new_labels = np.uint8(new_labels)
        if image.shape[2] > 1:  # RGB channel
            new_labels = np.repeat(new_labels, axis=2, repeats=image.shape[2])
        transformed_image, transformed_label = self.transform['aug'](image, new_labels)
        if if_slice_data:
            transformed_label = transformed_label[0, :, :]
        transformed_image  = self.intensity_norm_fn(transformed_image)
        result_dict = {
            'image': transformed_image,
            'label': transformed_label
        }
        if self.keep_orig_image_label_pair:
            orig_image_tensor, orig_label_tensor = self.transform['norm'](orig_image, new_labels)
            if if_slice_data:
                orig_label_tensor = orig_label_tensor[0, :, :]
            orig_image_tensor  = self.intensity_norm_fn(orig_image_tensor)
            result_dict['origin_image'] = orig_image_tensor
            result_dict['origin_label'] = orig_label_tensor
        return result_dict

    def load_data(self, index):
        '''
        give a index to fetch a data package for one patient
        :return:
        data from a patient.
        class dict: {
        'image': ndarray,H*W*CH, CH =1, for gray images
        'label': ndaray, H*W
        '''
        assert len(self.patient_id_list) > 0, "no data found in the disk at {}".format(self.root_dir)
        index = index % self.datasize
        patient_id, slice_id = self.find_pid_slice_id(index)
        if self.debug:
            print(patient_id)
        image_3d, label_3d, sitkImage, sitkLabel = self.load_patientImage_from_nrrd(
            patient_id, new_spacing=self.new_spacing, normalize=self.normalize)
        
        max_id = image_3d.shape[0]
        id_list = list(np.arange(max_id))

        image = image_3d[slice_id]
        label = label_3d[slice_id]
        # remove slice w.o objects
        while True:
            image = image_3d[slice_id]
            label = label_3d[slice_id]
            if abs(np.sum(label) - 0) > 1e-4:
                break
            else:
                id_list.remove(slice_id)
                random.shuffle(id_list)
                slice_id = id_list[0]

        if self.smooth_label:
            raise NotImplementedError
        ## add binary segmentation option
        if self.binary_segmentation:
            label[label > 0] = 1
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        if self.debug:
            print(image.shape)
            print(label.shape)
        self.pid = patient_id
        cur_data_dict = {'image': image,
                         'label': label,
                         'pid': patient_id,
                         'spacing': sitkImage.GetSpacing(),
                        'new_spacing':self.new_spacing}
        del image_3d,label_3d,sitkImage,sitkLabel
        self.temp_data_dict = cur_data_dict
        return cur_data_dict
    
    
    def formulate_labels(self, label, foreground_only=False):
        origin_labels = label.copy()
        if foreground_only:
            origin_labels[origin_labels > 0] = 1
            return origin_labels
        old_cls_to_idx_dict = switch_kv_in_dict(self.idx2cls_dict)
        new_cls_to_idx_dict = switch_kv_in_dict(self.formalized_label_dict)
        new_labels = np.zeros_like(label, dtype=np.uint8)
        for key in new_cls_to_idx_dict.keys():
            old_label_value = old_cls_to_idx_dict[key]
            new_label_value = new_cls_to_idx_dict[key]
            new_labels[origin_labels == old_label_value] = new_label_value
        return new_labels

    def find_pid_slice_id(self, index):
        '''
        given an index, find the patient id and slice id
        return the current id
        :return:
        '''
        self.pid = self.index2pid_dict[index]
        self.slice_id = self.index2slice_dict[index]

        return self.pid, self.slice_id
    @staticmethod
    def get_all_image_array_from_datastet(dataset):
        image_arrays = np.array([data['image'].numpy().reshape(1, -1).squeeze() for i, data in enumerate(dataset)])
        return image_arrays

    @staticmethod
    def get_mean_image(dataset):
        image_arrays = np.array([data['image'].numpy().reshape(1, -1).squeeze() for i, data in enumerate(dataset)])
        return np.mean(image_arrays, axis=0)

    
    def get_patient_data_for_testing(self, pid_index, crop_size=None, new_spacing=None, normalize_2D=True):
        '''
        prepare test volumetric data
        :param pad_size:[H',W']
        :param crop_size: [H',W']
        :return:
        data dict:
        {'image':torch tensor data N*1*H'*W'
        'label': torch tensor data: N*H'*W'
        }
        '''
        # print('here')
        if crop_size is None: crop_size = self.crop_size
        if new_spacing is None: new_spacing =  self.new_spacing
        self.pid = self.patient_id_list[pid_index]

        image, label, sitkImage, sitkLabel = self.load_patientImage_from_nrrd(
            self.pid,new_spacing=new_spacing, normalize=self.normalize) ## 3D subject level normalization
        ## update voxel spacing here
        if new_spacing is None:
            self.voxel_spacing = sitkImage.GetSpacing()
        if crop_size is not None:
            image, label, h_s, w_s, h, w = crop_or_pad(image, crop_size, label=label)
        image_tensor = torch.from_numpy(image[:, np.newaxis, :, :]).float()
        label_tensor = torch.from_numpy(label[:, :, :]).long()
        image_tensor = image_tensor.contiguous()
        if normalize_2D: image_tensor = self.intensity_norm_fn(image_tensor)
    
        return {
            'image': image_tensor,
            'label': label_tensor,
            'pid': self.pid,
            'new_spacing':self.voxel_spacing}

    def get_id(self):
        '''
        return the current patient id
        :return:
        '''
        return str(self.pid)

    def get_info(self):
        print('{} contains {} images with size of {}, num_classes: {} '.format(self.dataset_name, str(self.datasize),
                                                                               str(self.image_size),
                                                                               str(self.num_classes)))

    def save_cache(a_dict, cache_path):
        '''
        given a dict, save it to disk pth
        '''
        pass

    def load_cache(a_dict, cache_path):
        pass


class CombinedDataSet(data.Dataset):
    """
    source_dataset and augmented_source_dataset must be aligned
    """

    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __getitem__(self, index):
        source_index = index % len(self.source_dataset)
        target_index = (index + np.random.randint(0, len(self.target_dataset) - 1)) % len(self.target_dataset)

        return self.source_dataset[source_index], self.target_dataset[target_index]

    def __len__(self):
        return min(len(self.source_dataset), len(self.target_dataset))


class ConcatDataSet(data.Dataset):
    """
    concat a list of datasets together
    """

    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        a_sum = 0
        self.patient_number = 0
        self.formalized_label_dict = self.dataset_list[0].formalized_label_dict
        self.pid2datasetid = {}
        self.slice2datasetid = {}
        for dataset_id, dset in enumerate(self.dataset_list):
            for id in range(self.patient_number, self.patient_number + dset.patient_number):
                self.pid2datasetid[id] = dataset_id
            for sid in range(a_sum, a_sum + len(dset)):
                self.slice2datasetid[sid] = dataset_id
            a_sum += len(dset)
            self.patient_number += dset.patient_number
        self.datasize = a_sum
        print(f'total patient number: {self.patient_number}, 2D slice number:{self.datasize}')

    def __getitem__(self, index):
        dataset_id = self.slice2datasetid[index]
        if dataset_id >= 1:
            start_index = 0
            for ds in self.dataset_list[:dataset_id]:
                start_index += len(ds)
            index = index - start_index
        # print(f'index {index} dataset id {dataset_id}')
        self.cur_dataset = self.dataset_list[dataset_id]
        return self.cur_dataset[index]

    def __len__(self):
        return self.datasize

    def get_id(self):
        '''
        return the current patient id
        :return:
        '''

        return self.cur_dataset.get_id()

    def get_voxel_spacing(self):
        return self.cur_dataset.get_voxel_spacing()

    def get_patient_data_for_testing(self, pid_index, crop_size=None,new_spacing=None, normalize_2D=False):
        self.pid = pid_index
        dataset_id = self.pid2datasetid[pid_index]
        self.cur_dataset = self.dataset_list[dataset_id]
        index = pid_index % self.cur_dataset.patient_number
        data_pack = self.cur_dataset.get_patient_data_for_testing(index, crop_size, new_spacing,normalize_2D)
        return data_pack


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from medseg.dataset_loader.transform import Transformations  #
    from torch.utils.data import DataLoader

    image_size = (5, 5, 1)
    label_size = (5, 5)
    crop_size = (5, 5, 1)
    # class_dict={
    #   0: 'BG',  1: 'FG'}
    tr = Transformations(data_aug_policy_name='affine', crop_size=crop_size).get_transformation()
    dataset = BaseSegDataset(dataset_name='dummy', image_size=image_size, label_size=label_size, transform=tr['train'],
                             use_cache=True)
    dataset_2 = BaseSegDataset(dataset_name='dummy', image_size=image_size, label_size=label_size, transform=tr['train'],
                               use_cache=True)
    combined_train_loader = CombinedDataSet(source_dataset=dataset, target_dataset=dataset_2)
    train_loader = DataLoader(dataset=combined_train_loader, num_workers=0, batch_size=1, shuffle=True, drop_last=True)

    for i, item in enumerate(train_loader):
        source_input, target_input = item
        # print (source_input)
        img = source_input['image']
        label = target_input['origin_image']
        print(img.numpy().shape)
        print(label.numpy().shape)
        plt.subplot(121)
        plt.imshow(img.numpy()[0, 0])
        plt.subplot(122)
        plt.imshow(label.numpy()[0, 0])
        plt.colorbar()
        plt.show()
        break
