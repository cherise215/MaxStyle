# Created by cc215 at 02/05/19
# Modified by cc215 at 11/12/19

# This code is for testing basic segmentation networks
# Steps:
#  1. get the segmentation network and the path of checkpoint
#  2. fetch images tuples from the disk to test the segmentation
#  3. get the prediction result
#  4. update the metric
#  5. save the results.
from __future__ import print_function
from os.path import join
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import torch.nn.functional as F
import sys
sys.path.append('.')
from src.models.model_util import makeVariable
from src.models.base_segmentation_model import SegmentationModel
from src.dataset_loader.base_segmentation_dataset import BaseSegDataset
from src.common_utils.metrics import runningMySegmentationScore
from src.common_utils.save import save_nrrd_to_disk


class TestSegmentationNetwork():
    def __init__(self, test_dataset: BaseSegDataset, crop_size, segmentation_model, maximum_batch_size=1, use_gpu=True, save_path='',
                 summary_report_file_name='result.csv', detailed_report_file_name='details.csv',
                 save_prediction=False, patient_wise=False, metrics_list=['Dice', 'HD'],
                 normalize_2D=False, new_spacing=None,foreground_only=False, save_soft_prediction=False, sample_ratio_for_testing=None):
        '''
        perform segmentation model evaluation
        :param test_dataset: test_dataset
        :param segmentation_model: trained_segmentation_model
        '''
        self.test_dataset = test_dataset
        self.testdataloader = DataLoader(dataset=self.test_dataset, num_workers=0, batch_size=maximum_batch_size, shuffle=False,
                                         drop_last=False)

        self.segmentation_model = segmentation_model
        self.use_gpu = use_gpu
        self.num_classes = segmentation_model.num_classes

        self.segmentation_metric = runningMySegmentationScore(n_classes=segmentation_model.num_classes,
                                                              idx2cls_dict=self.test_dataset.formalized_label_dict,
                                                              metrics_list=metrics_list, foreground_only=foreground_only)
        self.sample_ratio_for_testing = sample_ratio_for_testing
        self.save_path = save_path
        self.summary_report_file_name = summary_report_file_name
        self.detailed_report_file_name = detailed_report_file_name
        self.normalize_2D = normalize_2D
        self.new_spacing = new_spacing
    
        self.crop_size = crop_size
        self.maximum_batch_size = maximum_batch_size
        self.save_prediction = save_prediction
        self.save_format_name = '{}_pred.npy'  # id plu
        self.patient_wise = patient_wise
        self.save_soft_prediction = save_soft_prediction
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.df = None
        self.result_dict = {}

    def run(self,):
        print('start evaluating')
        self.progress_bar = tqdm(total=100)

        if self.patient_wise:
            if self.sample_ratio_for_testing is not None:
                assert self.sample_ratio_for_testing <= 1.0, 'ratio for subset sampling must be below 1.0'
                random.seed('123456')
                n_test = int(np.round(self.sample_ratio_for_testing * self.test_dataset.patient_number, 0))
                n_test = 1 if n_test <= 0 else n_test
                if n_test < self.test_dataset.patient_number:
                    full_id_list = list(np.arange(self.test_dataset.patient_number))
                    id_list = random.sample(full_id_list, n_test)
                    print(id_list)
                else:
                    id_list = np.arange(self.test_dataset.patient_number)

                print(f'select {n_test} to evaluate')
            else:
                id_list = np.arange(self.test_dataset.patient_number)

            for i in id_list:
                data_tensor_pack = self.test_dataset.get_patient_data_for_testing(
                    i, crop_size=self.crop_size, new_spacing = self.new_spacing,normalize_2D=self.normalize_2D)

                pid, patient_triplet_result = self.evaluate(
                    i, data_tensor_pack, len(id_list), self.maximum_batch_size)
                pid = str(pid)
                self.result_dict[pid] = patient_triplet_result
        else:
            loader = self.testdataloader
            if self.sample_ratio_for_testing is not None:
                assert self.sample_ratio_for_testing <= 1.0, 'ratio for subset sampling must be below 1.0'
                random.seed('123456')
                n_test = int(np.round(self.sample_ratio_for_testing * len(loader), 0))
                n_test = 1 if n_test <= 0 else n_test
                id_list = random.sample(list(np.arange(len(loader))), n_test)
                print(f'select {n_test} to evaluate')

            else:
                id_list = np.arange(len(loader))
            for i, data_tensor_pack in enumerate(loader):
                if i in id_list:
                    pid, patient_triplet_result = self.evaluate(
                        i, data_tensor_pack, len(loader), self.maximum_batch_size)
                    pid = str(pid)
                    self.result_dict[pid] = patient_triplet_result

            # self.segmentation_model.save_testing_images_results(self.save_path, '', max_slices=10,file_name='{}.png'.format(pid))
        self.segmentation_metric.get_scores(save_path=join(self.save_path, self.summary_report_file_name))
        self.df = self.segmentation_metric.save_patient_wise_result_to_csv(
            save_path=join(self.save_path, self.detailed_report_file_name))
        # save top k and worst k cases
        print('<-finish->')

    def evaluate(self, i: int, data_tensor_pack: dict, total_number: int, maximum_batch_size=15):
        '''
        :param i: id
        :param data_tensor_pack:
        :return:
        '''
        assert maximum_batch_size > 0
        image = data_tensor_pack['image']
        label_npy = data_tensor_pack['label'].numpy()
        pid = self.test_dataset.get_id()
        pid = pid + ""
        total_size = image.size(0)
        # print(f'max batch size:{maximum_batch_size},got {total_size}')

        if total_size > maximum_batch_size:

            # image size is too large, break it down to chunks.[[start_id,end_id],[start_id,end_id]]
            a_split_ids = [[x, min(total_size, x + maximum_batch_size)]
                           for x in range(0, total_size, maximum_batch_size)]
        else:
            a_split_ids = [[0, total_size]]
        pred_npy = np.zeros_like(label_npy, dtype=np.uint8)
        soft_pred_npy = np.zeros_like(label_npy, dtype=np.float32)
        soft_pred_npy = soft_pred_npy[:, np.newaxis, :]
        soft_pred_npy = np.repeat(soft_pred_npy, repeats=self.num_classes, axis=1)

        for chunk_id in a_split_ids:
            image_a = image[chunk_id[0]:chunk_id[1], :, :, :]
            image_V_a = makeVariable(image_a, type='float', use_gpu=self.use_gpu, requires_grad=False)
            predict_a = self.segmentation_model.predict(input=image_V_a, softmax=False)

            pred_npy[chunk_id[0]:chunk_id[1]] = predict_a.max(1)[1].cpu().numpy()
            soft_pred_npy[chunk_id[0]:chunk_id[1]] = predict_a.data.cpu().numpy()

        # update metrics patient by patient
        # print (f'predict size: {pred_npy.shape}')
        
        self.segmentation_metric.update(pid=pid, preds=pred_npy, gts=label_npy,
                                        voxel_spacing=self.test_dataset.get_voxel_spacing())
        image_width = pred_npy.shape[-2]
        image_height = pred_npy.shape[-1]
        rgb_channel = image.size(1)

        assert rgb_channel == 1, 'currently only support gray images, found: {}'.format(rgb_channel)
        if label_npy.shape[0] == 1:
            # 2D images
            image_gt_pred = {
                'image': image.numpy().reshape(image_height, image_width),
                'label': label_npy.reshape(image_height, image_width),
                'pred': pred_npy.reshape(image_height, image_width),
                'soft_pred': soft_pred_npy.reshape(self.num_classes, image_height, image_width)
            }
        else:
            # 3D images

            image_gt_pred = {
                'image': image.numpy().reshape(-1, image_height, image_width),
                'label': label_npy.reshape(-1, image_height, image_width),
                'pred': pred_npy.reshape(-1, image_height, image_width),
                'soft_pred': soft_pred_npy.reshape(-1, self.num_classes, image_height, image_width),
            }

        self.progress_bar.update(100 * (i / total_number))
        # print('completed {cur_id}/{total_number}'.format(cur_id=str(i + 1), total_number=str(len(self.test_dataset))))

        if self.save_prediction:
            nrrd_save_path = os.path.join(self.save_path, 'pred_nrrd')
            if not os.path.exists(nrrd_save_path):
                os.makedirs(nrrd_save_path)
            image = image_gt_pred['image']
            pred = image_gt_pred['pred']
            gt = image_gt_pred['label']
            if '/' in pid:
                pid = pid.replace('/', '_')
            save_nrrd_to_disk(save_folder=nrrd_save_path, file_name=pid, image=image, pred=pred, gt=gt)
            print('save to:{}'.format(nrrd_save_path))
        if self.save_soft_prediction:
            npy_save_path = os.path.join(self.save_path, 'pred_npy')
            if not os.path.exists(npy_save_path):
                os.makedirs(npy_save_path)
            # save image and label and softprediction to numpy array
            if '/' in pid:
                pid = pid.replace('/', '_')
            save_path = join(npy_save_path, '{}_soft_pred.npy'.format(str(pid)))

            with open(save_path, 'wb') as f:
                np.save(file=save_path, arr=image_gt_pred['soft_pred'])

            # save_path = join(npy_save_path,'{}_hidden_feature.npy'.format(str(pid)))
            # with open(save_path, 'wb') as f:
            #     np.save(file = save_path, arr = image_gt_pred['hidden_feature'])

            save_path = join(npy_save_path, '{}_gt.npy'.format(str(pid)))
            with open(save_path, 'wb') as f:
                np.save(file=save_path, arr=image_gt_pred['label'])

            save_path = join(npy_save_path, '{}_image.npy'.format(str(pid)))
            with open(save_path, 'wb') as f:
                np.save(file=save_path, arr=image_gt_pred['image'])

        return pid, image_gt_pred

    def get_top_k_results(self, topk: int = 5, attribute: str = 'MYO_Dice', order: int = 0):
        '''
        select top k or worst k id according to the evaluation results,
        :param topk: number of k images
        :param attributes: attribute:classname+'_'+metric name, e.g: MYO_Dice
        :param order: the order for ranking. 0 (descending), 1 (ascending),
        :return: none
        '''
        assert not self.df is None and not self.result_dict is None, 'please run evaluation before saving'
        if order == 0:
            filtered_df = self.df.nlargest(topk, attribute)
        elif order == 1:
            filtered_df = self.df.nsmallest(topk, attribute)
        else:
            raise ValueError
        # get_patient_id
        print(filtered_df)
        return filtered_df


def save_top_k_result(filtered_df: pandas.DataFrame, result_dict: dict, attribute: str, file_format_name=None, save_path=None, save_nrrd=False):
    '''
    save top k results of (image, label, pred) to the disk

    :param filtered_df: the data frame after filtering.
    :param result_dict: the dict produced by the tester, which are triplets of image-gt-prediction results.
    :param attribute: the attribute (segmentation score) used to rank the segmentation results.
    :param file_format_name: the name format of each file. e.g. if 'pred_{}', then it will save each images as pred_{#id}.png.
    :param save_path: the directory for saving the results.
    :return:

    '''
    assert not save_path is None, 'save path can not be none'
    for id in filtered_df['patient_id'].values:
        if file_format_name is None:
            file_name = id
        else:
            file_name = file_format_name.format(id)
        image_gt_pred_triplet = result_dict[id]
        # save npy
        npy_save_path = os.path.join(save_path, 'pred_npy')
        if not os.path.exists(npy_save_path):
            os.makedirs(npy_save_path)
        np.save(os.path.join(npy_save_path, file_name + ".npy"), image_gt_pred_triplet)

        # save image
        image_save_path = os.path.join(save_path, 'pred_image')
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        if len(image_gt_pred_triplet['image'].shape) == 3:
            for ind in range(image_gt_pred_triplet['image'].shape[0]):
                paired_image = np.concatenate(
                    (image_gt_pred_triplet['image'][ind], image_gt_pred_triplet['label'][ind], image_gt_pred_triplet['pred'][ind]), axis=1)
                plt.imshow(paired_image, cmap="gray")
                plt.title("{id}:{attribute}{score:.2f}".format(id=id, attribute=attribute,
                                                               score=filtered_df[filtered_df['patient_id'] == id][
                                                                   attribute].values[0]))
                plt.savefig(os.path.join(image_save_path, file_name + "_" + str(ind) + ".png"))

        else:
            paired_image = np.concatenate(
                (image_gt_pred_triplet['image'], image_gt_pred_triplet['label'],
                 image_gt_pred_triplet['pred']), axis=1)
            plt.imshow(paired_image, cmap="gray")
            plt.title("{id}:{attribute}{score:.2f}".format(id=id, attribute=attribute,
                                                           score=filtered_df[filtered_df['patient_id'] == id][attribute].values[0]))
            plt.savefig(os.path.join(image_save_path, file_name + ".png"))
        # save nrrd:
        if save_nrrd:
            nrrd_save_path = os.path.join(save_path, 'pred_nrrd')
            image = image_gt_pred_triplet['image']
            pred = image_gt_pred_triplet['pred']
            gt = image_gt_pred_triplet['label']
            save_nrrd_to_disk(save_folder=nrrd_save_path, file_name=file_name, image=image, pred=pred, gt=gt)
