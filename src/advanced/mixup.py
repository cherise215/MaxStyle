# Created by cc215 at 13/12/19
import numpy as np
import torch
import sys
sys.path.append('../')
from src.models.custom_loss import One_Hot, cross_entropy_2D


class MixUP():
    def __init__(self, alpha=0.4, preserve_order=False, use_gpu=False, opt=None):
        '''
        MIXUP training
        reference: https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
        paper: https://arxiv.org/abs/1710.09412 mixup: Beyond Empirical Risk Minimization

        :param alpha:
        :param use_gpu:  alpha: mixup interpolation coefficient, using beta distribution: (alpha, alpha)
        '''
        self.alpha = alpha
        self.preserve_order = preserve_order
        self.use_gpu = use_gpu
        self.opt = opt
        self._init_from_opt(opt)
        self.perm_index = None

    def _init_from_opt(self, opt):
        if opt is not None:
            if 'alpha' in opt.keys():
                self.alpha = opt['alpha']

    def get_mixup_data(self, x_train_batch, y_train_batch, mix_y=False, num_classes=None):
        '''
        code adapted from https://github.com/krishnabits001/task_driven_data_augmentation/blob/master/utils.py
        # Generator for mixup data - to linearly combine 2 random image,label pairs from the batch of image,label pairs
        input params:
            x_train: batch of input images tensor: N*C*H*W
            y_train: batch of input labels tensor: N*H*W (for segmentation task), N (for classification task)
            alpha: control mix-up interpolation distribution, for beta distribution: (alpha, alpha)
        returns:
            x_out: linearly combined resultant image
            y_a: original prediction y
            y_b: perturbed prediction y for mixing.
            lambda: mixup interpolation coefficient
        '''

        if self.alpha > 0:
            # sample coefficient
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            # do nothing
            lam = 1
        batch_size = x_train_batch.size()[0]
        if self.use_gpu:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
        if self.preserve_order:
            # higher weight to the first item
            lam = max([lam, 1 - lam])

        x_a = x_train_batch
        x_b = x_train_batch[index, :]
        mixed_x = lam * x_a + (1 - lam) * x_b
        self.perm_index = index
        if mix_y:
            assert num_classes is not None and isinstance(num_classes, int), "number of classes must be provided"
            y_map = One_Hot(num_classes, use_gpu=self.use_gpu)(y_train_batch)
            mixed_y = lam * y_map + (1 - lam) * (y_map[index])
            return mixed_x, mixed_y
        else:
            y_a = y_train_batch
            y_b = y_train_batch[index]
            return mixed_x, y_a, y_b, lam

    def get_mixup_loss(self, loss_fn, pred, y_a, y_b, lam: float):
        '''
        get the loss of training with mixed inputs for optimization

        :param loss_fn: the loss function
        :param pred: prediction from the classfication/segmentation model
        :param y_a: original batch of y
        :param y_b: perturbed batch of y for pairing
        :param lam: floatPweight for linear interpolation
        :return:
        '''
        return lam * loss_fn(pred, y_a) + (1 - lam) * loss_fn(pred, y_b)


class ManifoldMixup(MixUP):
    """[summary]
     apply mixup to intermediate features in a neural network
    """

    def __init__(self, alpha=2.0, preserve_order=False, use_gpu=False, opt=None):
        super(ManifoldMixup, self).__init__(alpha=alpha, preserve_order=preserve_order, use_gpu=use_gpu, opt=opt)
        self.lam = None
        self.perm_index = None

    def get_mixup_data(self, x_train_batch, y_train_batch=None, mix_y=True, num_classes=None):
        B = x_train_batch.size(0)
        if self.perm_index is None:
            self.perm_index = torch.randperm(B).to(x_train_batch.device)
        if self.lam is None:
            if self.alpha > 0:
                # sample coefficient
                self.lam = np.random.beta(self.alpha, self.alpha)

                if self.preserve_order:
                    # higher weight to the first item
                    self.lam = max([self.lam, 1 - self.lam])
            else:
                # do nothing
                self.lam = 1

        mixed_x = self.lam * x_train_batch + (1 - self.lam) * x_train_batch[self.perm_index]
        if y_train_batch is not None:
            if mix_y:
                assert num_classes is not None and isinstance(num_classes, int), "number of classes must be provided"
                y_map = One_Hot(num_classes, use_gpu=self.use_gpu)(y_train_batch)
                mixed_y = self.lam * y_map + (1 - self.lam) * (y_map[self.perm_index])
                return mixed_x, mixed_y
            else:
                y_a = y_train_batch
                y_b = y_train_batch[self.perm_index]
                return mixed_x, y_a, y_b
        else:
            return mixed_x, None
