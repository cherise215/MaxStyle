# Created by cc215 at 02/05/19
# segmentation model definition goes here

import os
from os.path import join
import torch.nn as nn
import torch
import torch.optim as optim
import gc
import traceback
import sys
sys.path.append('../')

from src.models.init_weight import init_weights
from src.models.segmentation_models.fcn import FCN
from src.models.segmentation_models.unet import UNet, UNetv2
from src.models.segmentation_models.resconvunet import ResConvUNet
from src.models.model_util import ExponentialMovingAverage
from src.models.model_util import get_scheduler
from src.common_utils.metrics import runningScore
from src.common_utils.save import save_testing_images_results


class SegmentationModel(nn.Module):
    def __init__(self, network_type, in_channels=1, num_classes=2,
                 encoder_dropout=None,
                 decoder_dropout=None, use_gpu=True, lr=0.001, resume_path=None,
                 use_ema=False,
                 optimizer='Adam'
                 ):
        '''
        return a segmentation model
        :param network_type: string
        :param num_domains: int
        :param in_channels: int
        :param num_classes: int
        :param encoder_dropout: float
        :param decoder_dropout: float
        :param use_gpu: bool
        :param lr: float
        :param resume_path: string
        :param loss_type: string: loss for segmentation, support multiple losses with '+' as connection, e.g. 'cross_entropy+dice'
        :param loss_term_weights: list: weighted loss for each loss term,support multi-loss terms e.g. '0.5,0.5'
        '''
        self.network_type = network_type
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.lr = lr
        self.in_channels = in_channels
        self.encoder_dropout = encoder_dropout if isinstance(encoder_dropout, float) else None
        self.decoder_dropout = decoder_dropout if isinstance(encoder_dropout, float) else None

        self.model = self.get_network_from_model_library(self.network_type)
        assert not self.model is None, 'cannot find the model given the specified name'
        # print number of paramters
        self.resume_path = resume_path
        self.init_model(network_type)
        if self.use_gpu:
            self.model.cuda()

        self.set_optmizers(name=optimizer)
        self.use_ema = use_ema
        if use_ema:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.999)
        else:
            self.ema = None

        self.running_metric = self.set_running_metric()  # cal iou score during training

        self.cur_eval_images = None
        self.cur_eval_predicts = None
        self.cur_eval_gts = None  # N*H*W

        # template checkpoint path
        self.save_rescue_postfix = join(
            'last', *['checkpoints', self.network_type + '$' + 'SAX' + '$' + '_Segmentation' + '.pkl'])

        self.loss = 0.

    def get_network_from_model_library(self, network_type):
        model = None
        if network_type == 'UNet_16':
            model = UNet(input_channel=self.in_channels, num_classes=self.num_classes, feature_scale=4,
                         norm=nn.BatchNorm2d, if_SN=False,
                         self_attention=False, encoder_dropout=self.encoder_dropout,
                         decoder_dropout=self.decoder_dropout)
            print('init {}'.format(network_type))
        elif network_type == 'UNetv2_16':
            model = UNetv2(input_channel=self.in_channels, num_classes=self.num_classes, feature_scale=4,
                           norm=nn.BatchNorm2d, if_SN=False,
                           self_attention=False, encoder_dropout=self.encoder_dropout,
                           decoder_dropout=self.decoder_dropout)
            print('init {}'.format(network_type))
        elif network_type == 'UNet_32':
            model = UNet(input_channel=self.in_channels, num_classes=self.num_classes, feature_scale=2,
                         norm=nn.BatchNorm2d, if_SN=False,
                         self_attention=False, encoder_dropout=self.encoder_dropout,
                         decoder_dropout=self.decoder_dropout)
            print('init {}'.format(network_type))

        elif network_type == 'UNet_64':
            model = UNet(input_channel=self.in_channels, num_classes=self.num_classes, feature_scale=1,
                         norm=nn.BatchNorm2d, if_SN=False,
                         self_attention=False, encoder_dropout=self.encoder_dropout,
                         decoder_dropout=self.decoder_dropout)
            print('init {}'.format(network_type))
        elif network_type == 'SN_UNet_16':
            model = UNet(input_channel=self.in_channels, num_classes=self.num_classes, feature_scale=4,
                         norm=nn.BatchNorm2d, if_SN=True,
                         self_attention=False, encoder_dropout=self.encoder_dropout,
                         decoder_dropout=self.decoder_dropout)
            print('init {}'.format(network_type))
        elif network_type == 'IN_SN_UNet_16':
            model = UNet(input_channel=self.in_channels, num_classes=self.num_classes, feature_scale=4,
                         norm=nn.InstanceNorm2d, if_SN=True,
                         self_attention=False, encoder_dropout=self.encoder_dropout,
                         decoder_dropout=self.decoder_dropout)
            print('init {}'.format(network_type))
        elif network_type == 'FCN_16':
            model = FCN(input_channel=self.in_channels, num_classes=self.num_classes, feature_scale=4,
                        decoder_dropout=self.decoder_dropout)
            print('init {}'.format(network_type))
        elif network_type == 'FCN_64':
            model = FCN(input_channel=self.in_channels, num_classes=self.num_classes, feature_scale=1,
                        decoder_dropout=self.decoder_dropout)
            print('init {}'.format(network_type))
        elif network_type == 'ResUNet_64':
            model = ResConvUNet(input_channel=self.in_channels, num_classes=self.num_classes, feature_scale=1,
                                decoder_dropout=self.decoder_dropout, encoder_dropout=self.encoder_dropout)
            print('init {}'.format(network_type))
        elif network_type == 'ResUNet_16':
            model = ResConvUNet(input_channel=self.in_channels, num_classes=self.num_classes, feature_scale=4,
                                decoder_dropout=self.decoder_dropout, encoder_dropout=self.encoder_dropout)
            print('init {}'.format(network_type))
        else:
            print('currently, we only support network types: [UNet_16,SN_UNet_16,IN_SN_UNet_16,FCN_16], but found {}'.format(
                network_type))
            raise NotImplementedError

        return model

    def init_model(self, network_type):
        resume_path = self.resume_path
        loaded = False
        if not resume_path is None:
            if not resume_path == '':
                assert os.path.exists(resume_path), 'path: {} must exist'.format(resume_path)
                if 'pth' in resume_path:
                    try:
                        loaded = True
                        self.model.load_state_dict(torch.load(resume_path), strict=False)
                    except:
                        traceback.print_exc()
                        print('fail to load checkpoint under {}'.format(resume_path))
                        loaded = False
                elif 'pkl' in resume_path:
                    # code for resumed model.
                    try:
                        self.model.load_state_dict(torch.load(resume_path)['model_state'], strict=False)
                        loaded = True
                    except Exception:
                        traceback.print_exc()
                        print('fail to load checkpoint under {}'.format(resume_path))
                        loaded = False
            else:
                print('can not find checkpoint under {}'.format(resume_path))
                loaded = False

        if not loaded:
            if 'FCN' in network_type:
                self.model.init_weights()
            else:
                init_weights(self.model, init_type='kaiming')
            print('randomly init ', network_type)
        else:
            print('loaded weights from path: {}'.format(resume_path))

    def forward(self, input):
        pred = self.model.forward(input)
        return pred

    def eval(self):
        self.model.eval()

    def get_loss(self, pred, targets=None, class_weights=None, loss_type='cross_entropy'):
        if not targets is None:
            loss = self.basic_loss_fn(pred, targets, class_weights=class_weights, loss_type=loss_type)
        else:
            loss = 0.
        self.loss = loss
        return self.loss

    def get_teacher_model(self):
        if self.use_ema:
            # First save original parameters before replacing with EMA version
            self.ema.store(self.model.parameters())
            # Copy EMA parameters to model
            self.ema.copy_to(self.model.parameters())
        return self.model

    def get_student_model(self):
        if self.use_ema:
            self.ema.restore(self.model.parameters())
        return self.model

    def train(self, if_testing=False):
        if not if_testing:
            self.model.train()
        else:
            self.model.eval()

    def reset_optimizers(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        # when set_to_none is true,it can lower memory footprint, https://pytorch.org/docs/master/optim.html#torch.optim.Optimizer.zero_grad

    def set_optmizers(self, name='Adam'):
        if name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = None
        elif name == 'AdaAdam':
            # with decreased learning rate
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = get_scheduler(self.optimizer, lr_policy='step', lr_decay_iters=50)
        else:
            raise NotImplementedError

    def optimize_params(self):
        self.optimizer.step()
        if self.use_ema:
            self.ema.update(self.model.parameters())

    def reset_loss(self):
        self.loss = 0.

    def set_running_metric(self):
        running_metric = runningScore(n_classes=self.num_classes)
        return running_metric

    def predict(self, input, softmax=True):
        gc.collect()  # collect garbage
        self.train(if_testing=True)
        with torch.no_grad():
            output = self.model.forward(input)
        if softmax:
            output = torch.nn.Softmax2d()(output)
        return output

    def evaluate(self, input, targets_npy):
        '''
        evaluate the model performance

        :param input: 4-d tensor input: NCHW
        :param targets_npy: numpy ndarray: N*H*W
        :param running_metric: runnning metric for evaluatation
        :return: 3d segmentation maps (numpy 3d-array) 
        '''
        gc.collect()  # collect garbage
        self.train(if_testing=True)
        pred = self.predict(input)
        pred_npy = pred.max(1)[1].cpu().numpy()
        del pred
        self.running_metric.update(label_trues=targets_npy, label_preds=pred_npy)
        self.cur_eval_images = input.data.cpu().numpy()[:, 0, :, :]
        del input
        self.cur_eval_predicts = pred_npy
        self.cur_eval_gts = targets_npy  # N*H*W

        return pred_npy

    def set_model_grad(self, state=False):
        assert self.model
        for p in self.model.parameters():  # reset requires_grad
            p.requires_grad = state

    def save_model(self, save_dir, epoch_iter, model_prefix=None):
        if model_prefix is None:
            model_prefix = self.network_type
        epoch_path = join(save_dir, *[str(epoch_iter), 'checkpoints'])
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)

        torch.save(self.model.state_dict(),
                   join(epoch_path, model_prefix + '$' + 'SAX' + '$' + '_Segmentation' + '.pth'))

    def load_snapshots(self, file_path):
        start_epoch = 0
        try:
            checkpoint = torch.load(file_path)
            for k, v in checkpoint.items():
                if k == 'model_state':
                    state_dict = checkpoint['model_state']
                    self.model.load_state_dict(state_dict, strict=False)

                if k == 'optimizer_state':
                    optimizer_state = checkpoint['optimizer_state']
                    if not (optimizer_state is None) and (not self.optimizer is None):
                        try:
                            self.optimizer.load_state_dict(optimizer_state)
                        except:
                            pass
                if k == 'epoch':
                    start_epoch = int(checkpoint['epoch'])
            print("Loaded checkpoint '{}' (epoch {})".format(file_path, checkpoint['epoch']))
        except:
            print('{} for restoring is not found'.format(file_path))
        return start_epoch

    def save_snapshots(self, epoch, save_dir, prefix=None):

        if prefix is None:
            prefix = epoch
        epoch_path = join(save_dir, *[str(prefix), 'checkpoints'])
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)
        save_path = join(epoch_path, self.network_type + '$' + 'SAX' + '$' + '_Segmentation' + '.pkl')
        state_dict = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()
        state = {'network_type': self.network_type,
                 'epoch': epoch,
                 'model_state': state_dict,
                 'optimizer_state': self.optimizer.state_dict()
                 }
        torch.save(state, save_path)

    def save_current_results(self, save_name='predict.npy'):
        raise NotImplementedError

    def save_testing_images_results(self, save_dir, epoch_iter, max_slices=10, file_name='Seg_plots.png'):
        gts = self.cur_eval_gts
        predicts = self.cur_eval_predicts
        images = self.cur_eval_images
        save_testing_images_results(images, gts, predicts, save_dir, epoch_iter, max_slices, file_name)
