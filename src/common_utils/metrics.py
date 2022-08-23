# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
from medpy.metric.binary import dc
import pandas as pd
from IPython.display import display, HTML

from src.common_utils.measure import hd, hd_2D_stack, asd, volumesimilarity, VolumeSimIndex


class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall Acc: \t': acc,
                'Mean Acc : \t': acc_cls,
                'FreqW Acc : \t': fwavacc,
                'Mean IoU : \t': mean_iu, }, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class runningCustomScore(object):

    def __init__(self, n_classes, add_hd=False):
        self.n_classes = n_classes
        assert self.n_classes <= 2, 'only support binary segmentation for now'
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.dice_score = []
        self.hd_score = []
        self.add_hd = add_hd

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds, voxel_spacing=None):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
            # Clip the value to compute the volumes
            gt = np.clip(label_trues, 0, 1)
            pred = np.clip(label_preds, 0, 1)
            self.dice_score.append(dc(result=pred, reference=gt))
            if self.add_hd:
                assert voxel_spacing is not None, 'please define voxel '
                if np.sum(gt) > 0 and np.sum(pred) > 0:
                    print(voxel_spacing)
                    self.hd_score.append(hd(result=pred, reference=gt, voxelspacing=voxel_spacing, connectivity=1))

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        mean_dice = np.mean(self.dice_score)
        std_dice = np.std(self.dice_score)
        if self.add_hd:
            mean_hd = np.mean(self.hd_score)
            std_hd = np.std(self.hd_score)

            return {'Overall Acc: \t': acc,
                    'Mean Acc : \t': acc_cls,
                    'FreqW Acc : \t': fwavacc,
                    'Mean IoU : \t': mean_iu,
                    'Mean Dice: \t': mean_dice,
                    'Std Dice: \t': std_dice,
                    'Mean HD: \t': mean_hd,
                    'Std HD: \t': std_hd
                    }, cls_iu
        else:

            return {'Overall Acc: \t': acc,
                    'Mean Acc : \t': acc_cls,
                    'FreqW Acc : \t': fwavacc,
                    'Mean IoU : \t': mean_iu,
                    'Mean Dice: \t': mean_dice,
                    'Std Dice: \t': std_dice,
                    # 'Mean HD: \t': mean_hd,
                    # 'Std HD: \t': std_hd
                    }, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.dice_score = []
        self.hd_score = []


class runningMySegmentationScore(object):
    # segmentation metrics for 3D prediction (multi_classes)

    def __init__(self, n_classes, idx2cls_dict=None, metrics_list=['Dice'], foreground_only=False):
        """[summary]

        Args:
            n_classes ([type]): [description]
            idx2cls_dict ([type], optional): [description]. Defaults to None.
            metrics_list (list, optional): [description]. Defaults to ['Dice'].
            foreground_only (bool, optional): if true, will treat it as a binary segmentation task for performance measurement. Defaults to False.
        """
        self.n_classes = n_classes
        self.metrics = metrics_list
        self.multi_scores = {}
        self.tables = []
        self.foreground_only = foreground_only
        self.idx2cls_dict = idx2cls_dict
        if idx2cls_dict is None:
            self.idx2cls_dict = {}
            if self.foreground_only:
                self.idx2cls_dict[1] = 'foreground'
            else:
                for cls_id in range(n_classes):
                    self.idx2cls_dict[cls_id] = str(cls_id)
        support_metrics = ['Dice', 'HD', 'ASD', 'VolError', 'VolSim']
        header = ['patient_id']
        for c_index, class_name in self.idx2cls_dict.items():
            if c_index > 0:
                for metrics_name in metrics_list:
                    assert metrics_name in support_metrics, 'make sure that the metric func name is correct and has been registered in support_metrics '
                    self.multi_scores[class_name + '_' + metrics_name] = []
                    header += [class_name + '_' + metrics_name]
        self.header = header

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def update(self, pid, preds, gts, voxel_spacing=None):
        '''

        :param pid: patient id for recording
        :param preds: ndarray [int]: 3D input: n_slices*H*W
        :param gts: ndarray [int]: 3D ground truth: n_slices*H*W
        :param voxel_spacing:
        :return:
        '''
        assert preds.shape == gts.shape, 'pid :{} shape not consistent: pred {} vs gt {}'.format(
            pid, preds.shape, gts.shape)
        if not voxel_spacing is None:
            assert len(voxel_spacing) == 3, 'check voxel spacing, {}'.format(voxel_spacing)
        else:
            # print ('use default voxel spacing with 1.0 along each direction')
            voxel_spacing = np.ones(gts.shape, dtype=np.float32)
        n, h, w = preds.shape

        one_patient_result = [str(pid)]
        for c, class_name in self.idx2cls_dict.items():
            if c == 0:
                continue
            # Copy the gt image to not alterate the input
            gt_c_i = np.copy(gts)
            if self.foreground_only:
                gt_c_i[gt_c_i > 0] = 1
            else:
                gt_c_i[gt_c_i != c] = 0

            # Copy the pred image to not alterate the input
            pred_c_i = np.copy(preds)
            if self.foreground_only:
                pred_c_i[pred_c_i > 0] = 1
            else:
                pred_c_i[pred_c_i != c] = 0

            # Clip the value to compute the volumes
            gt_c_i = np.clip(gt_c_i, 0, 1)
            pred_c_i = np.clip(pred_c_i, 0, 1)

            for metric in self.metrics:
                if metric == 'Dice':
                    score = dc(result=pred_c_i, reference=gt_c_i)

                if metric == 'HD':
                    # 2D stack hausdorff distance, ignore z space
                    # check
                    assert voxel_spacing is not None
                    if len(voxel_spacing) >= 2:
                        assert voxel_spacing[0] >= voxel_spacing[2], 'z spacing should be in last dim in the cardiac imaging'
                    score = hd_2D_stack(result=pred_c_i.reshape(n, h, w), reference=gt_c_i.reshape(n, h, w),
                                        pixelspacing=voxel_spacing[:2], connectivity=2)  # 8 neighborhodd

                if metric == 'ASD':
                    assert voxel_spacing is not None
                    score = asd(result=pred_c_i.reshape(n, h, w), reference=gt_c_i.reshape(n, h, w),
                                voxelspacing=voxel_spacing, connectivity=2)

                if metric == 'VolSim' and c > 0:
                    score = VolumeSimIndex(result=pred_c_i.reshape(n, h, w),
                                           reference=gt_c_i.reshape(n, h, w))

                if metric == 'VolError' and c > 0:
                    # (pred-gt)/gt
                    score = (np.count_nonzero(pred_c_i) - np.count_nonzero(gt_c_i)) / (1.0 * np.count_nonzero(gt_c_i))

                self.multi_scores[class_name + '_' + metric].append(score)
                one_patient_result += [score] if not isinstance(score, list) else score
        self.tables.append(one_patient_result)
        return one_patient_result

    def get_scores(self, save_path=None):
        """Returns mean and std score evaluation result.
        if save path is valid, save summary result to csv.
        """
        summary_dict = {}
        summary_list = [[], []]  # for csv
        header = []
        for k, result_list in self.multi_scores.items():
            mean = np.mean(result_list)
            std = np.std(result_list)
            summary_dict[k + '_mean'] = mean
            summary_dict[k + '_std'] = std
            mean = '{:.3f}'.format(mean)
            std = '{:.3f}'.format(std)

            summary_list[0] += [mean]
            summary_list[1] += [std]
            header.append(k)
  
        if save_path is not None:
            df = pd.DataFrame(summary_list, columns=header)
            df.to_csv(save_path, index=False)

        return summary_dict, summary_list, header

    def save_patient_wise_result_to_csv(self, save_path):
        '''
        save patient-wise records to csv
        :param save_path:
        :return:
        '''
        df = pd.DataFrame(self.tables, columns=self.header)
        print('save to', save_path)
        if not save_path is None:
            df.to_csv(save_path, index=False)
        return df

    def reset(self):
        for k, score in self.multi_scores.items():
            self.multi_scores[k] = []
        self.tables = []


def cal_cls_acc(pred, gt):
    '''
    input tensor
    :param pred: network output N*n_classes
    :param gt: ground_truth N [labels_id]
    :return: float acc
    '''
    pred = pred.clone()
    pred_class = pred.data.max(1)[1].cpu()
    sum = gt.eq(pred_class).sum()
    count = gt.size(0)
    return sum, count


class runningAPScore(object):

    def __init__(self, thresh_hold, imagesize):
        self.thresh_hold = 2  # if two center points are close to each other within the range of n pixels.
        self.image_size = imagesize
        self.d_records = []
        self.FP = 0.
        self.TP = 0.
        self.npos = 0

    def update(self, label_trues, label_preds, objective_trues, objective_preds):
        '''

        :param label_trues: location in image coordinates, N*2
        :objective_score: N
        :objective_preds: N
        :param label_preds: N*2
        :return:
        '''

        for lt, d_t, lp, d_p in zip(label_trues, objective_trues, label_preds, objective_preds):
            lt = np.array(lt)
            lt = lt * (self.image_size / 2.) + self.image_size / 2.
            lp = np.array(lp)
            lp = lp * (self.image_size / 2.) + self.image_size / 2.
            dy = int(np.abs(lt[0] - lp[0]))
            dx = int(np.abs(lt[1] - lp[1]))

            d_p = int(d_p)
            d_t = int(d_t)
            dist = dy ** 2 + dx ** 2
            if d_p == 1 and d_t == 1:
                if dist < self.thresh_hold ** 2 + self.thresh_hold ** 2:  # count as true positive
                    self.TP += 1  # TP
                else:
                    self.FP += 1  # FP
            elif d_p == 1 and d_t == 0:
                self.FP += 1  # FP

            if d_t == 1:
                self.npos += 1  # the number of gts

    def get_scores(self):
        prec = np.divide(self.TP, (self.FP + self.TP))
        recall = self.TP / (1.0 * self.npos)
        return {'precision': prec,
                'recall': recall,
                'num_TP': self.TP,
                'num_FP': self.FP,
                'total positives': self.npos
                }

    def reset(self):
        self.d_records = []
        self.FP = 0.
        self.TP = 0.
        self.npos = 0


def print_metric(running_metrics, name='A->A'):
    score, class_iou = running_metrics.get_scores()
    print(name + ' score:')
    for k, v in score.items():
        if k == 'Mean IoU : \t':
            print(k, v)
    return score


def write_eval_scores_to_disk(running_metrics_groups, txt_path, views):
    file = open(txt_path, 'w')
    header = []
    metrics = ['Dice', 'HD']
    for metric in metrics:
        for view_name in views:
            rec_name = view_name + ' [' + metric + '] '
            header.append(rec_name + ' , ')
    header.append('\n')
    file.writelines(header)
    i_j_dices = []
    for view_name in views:
        running_metrics = running_metrics_groups[view_name]
        score, class_iou = running_metrics.get_scores()
        mean_dice = score['Mean Dice: \t']
        std_dice = score['Std Dice: \t']
        dice_item = '{:.3f} ({:.3f}), '.format(mean_dice, std_dice)
        i_j_dices.append(dice_item)
    for view_name in views:
        running_metrics = running_metrics_groups[view_name]
        score, class_iou = running_metrics.get_scores()
        mean_hd = score['Mean HD: \t']
        std_hd = score['Std HD: \t']
        hd_item = '{:.3f} ({:.3f}), '.format(mean_hd, std_hd)
        i_j_dices.append(hd_item)
    i_j_dices.append('\n')
    file.writelines(i_j_dices)
    file.close()
