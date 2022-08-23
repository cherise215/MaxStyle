
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import sys
sys.path.append('../../')
from src.models.custom_layers import DomainSpecificBatchNorm2d
from src.models.custom_layers import Fixable2DDropout
from src.models.model_util import _disable_tracking_bn_stats


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class res_convdown(nn.Module):
    '''
    '''

    def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None):
        super(res_convdown, self).__init__()
        # down-> conv3->prelu->conv
        if if_SN:
            self.down = spectral_norm(nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, bias=bias))

            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias)),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias)),
                norm(out_ch),
            )
        else:
            self.down = (nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, bias=bias))

            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
            )
        if if_SN:
            self.conv_input = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias))
        else:
            self.conv_input = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

        self.last_act = nn.LeakyReLU(0.2)
        self.dropout = dropout
        if not self.dropout is None:
            # self.drop = nn.Dropout2d(p=dropout)
            self.drop= Fixable2DDropout(p=dropout)

    def get_features(self, x):
        x = self.down(x)
        res_x = self.conv_input(x) + self.conv(x)
        return res_x

    def non_linear(self, x):
        res_x = self.last_act(x)
        if not self.dropout is None:
            res_x = self.drop(res_x)
        return res_x

    def forward(self, x):
        x = self.get_features(x)
        x = self.non_linear(x)
        return x


# class res_convup(nn.Module):
#     '''
#     upscale
#     '''

#     def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None):
#         super(res_convup, self).__init__()
#         # up-> conv3->prelu->conv

#         if if_SN:
#             self.up = nn.Sequential(
#                 spectral_norm(nn.ConvTranspose2d(in_ch, in_ch, 4, padding=1, stride=2), dim=1),

#             )
#             self.conv = nn.Sequential(
#                 spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias), dim=1),
#                 norm(out_ch),
#                 nn.LeakyReLU(0.2),
#                 spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias), dim=1),
#                 norm(out_ch),
#             )
#         else:
#             self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, padding=1, stride=2)
#             self.conv = nn.Sequential(
#                 nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1, bias=bias),
#                 norm(out_ch),
#                 nn.LeakyReLU(0.2),
#                 nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
#                 norm(out_ch),
#             )

#         if if_SN:
#             self.conv_input = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1,
#                                                       stride=1, padding=0, bias=bias), dim=1)
#         else:
#             self.conv_input = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

#         self.last_act = nn.LeakyReLU(0.2)
#         self.dropout = dropout
#         if not self.dropout is None:
#             self.drop = nn.Dropout2d(p=dropout)

#     def forward(self, x):
#         x = self.up(x)
#         res_x = self.last_act(self.conv_input(x) + self.conv(x))
#         if not self.dropout is None:
#             res_x = self.drop(res_x)
#         # appl
#         return res_x


# class res_convup_2(nn.Module):
#     '''
#     upscale
#     '''

#     def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None):
#         super(res_convup_2, self).__init__()
#         # up-> conv3->prelu->conv

#         if if_SN:
#             self.up = nn.Sequential(
#                 spectral_norm(nn.ConvTranspose2d(in_ch, in_ch, 4, padding=1, stride=2), dim=1),

#             )
#             self.conv = nn.Sequential(
#                 spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias), dim=1),
#                 norm(out_ch),
#                 nn.LeakyReLU(0.2),
#                 spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias), dim=1),
#                 norm(out_ch),
#             )
#         else:
#             self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, padding=1, stride=2)
#             self.conv = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias),
#                 norm(out_ch),
#                 nn.LeakyReLU(0.2),
#                 nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
#                 norm(out_ch),
#             )

#         if if_SN:
#             self.conv_input = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1,
#                                                       stride=1, padding=0, bias=bias), dim=1)
#         else:
#             self.conv_input = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

#         self.last_act = nn.LeakyReLU(0.2)
#         self.dropout = dropout
#         if not self.dropout is None:
#             self.drop = nn.Dropout2d(p=dropout)

#     def forward(self, x):
#         x = self.up(x)
#         res_x = self.last_act(self.conv_input(x) + self.conv(x))
#         if not self.dropout is None:
#             res_x = self.drop(res_x)
#         # appl
#         return res_x


# class res_bilinear_up(nn.Module):
#     '''
#     upscale
#     '''

#     def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None):
#         super(res_bilinear_up, self).__init__()
#         # up-> conv3->prelu->conv

#         if if_SN:
#             self.up = nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor=2),
#                 nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=bias)
#             )

#             self.conv = nn.Sequential(
#                 spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias), dim=1),
#                 norm(out_ch),
#                 nn.LeakyReLU(0.2),
#                 spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias), dim=1),
#                 norm(out_ch),
#             )
#         else:
#             self.up = nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor=2),
#                 nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=bias)
#             )

#             self.conv = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias),
#                 norm(out_ch),
#                 nn.LeakyReLU(0.2),
#                 nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
#                 norm(out_ch),
#             )

#         if if_SN:
#             self.conv_input = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1,
#                                                       stride=1, padding=0, bias=bias), dim=1)
#         else:
#             self.conv_input = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

#         self.last_act = nn.LeakyReLU(0.2)
#         self.dropout = dropout
#         if not self.dropout is None:
#             self.drop = nn.Dropout2d(p=dropout)

#     def forward(self, x):
#         x = self.up(x)
#         res_x = self.last_act(self.conv_input(x) + self.conv(x))
#         if not self.dropout is None:
#             res_x = self.drop(res_x)
#         # appl
#         return res_x


class res_NN_up(nn.Module):
    '''
    upscale with NN upsampling followed by conv
    '''

    def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None):
        super(res_NN_up, self).__init__()
        # up-> conv3->prelu->conv
        self.up = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=bias)
        )

        if if_SN:

            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias), dim=1),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias), dim=1),
                norm(out_ch),
            )
        else:

            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
            )

        if if_SN:
            self.conv_input = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                                      stride=1, padding=0, bias=bias), dim=1)
        else:
            self.conv_input = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

        self.last_act = nn.LeakyReLU(0.2)
        self.dropout = dropout
        if not self.dropout is None:
            self.drop= Fixable2DDropout(p=dropout)

            # self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.up(x)
        res_x = self.last_act(self.conv_input(x) + self.conv(x))
        if not self.dropout is None:
            res_x = self.drop(res_x)
        # appl
        return res_x


class res_up_family(nn.Module):
    '''
    upscale with different upsampling methods
    '''

    def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None, up_type='bilinear'):
        super(res_up_family, self).__init__()
        # up-> conv3->prelu->conv
        if up_type == 'NN':
            self.up = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
            )
        elif up_type == 'bilinear':
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
            )
        elif up_type == 'Conv2':
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        elif up_type == 'Conv4':
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)
        else:
            raise NotImplementedError

        if if_SN:

            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias), dim=1),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias), dim=1),
                norm(out_ch),
            )
        else:

            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
            )

        if if_SN:
            self.conv_input = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                                      stride=1, padding=0, bias=bias), dim=1)
        else:
            self.conv_input = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

        self.last_act = nn.LeakyReLU(0.2)
        self.dropout = dropout
        if not self.dropout is None:
            self.drop= Fixable2DDropout(p=dropout)
            # self.drop = nn.Dropout2d(p=dropout)

    def get_features(self, x):
        x = self.up(x)
        res_x = self.conv_input(x) + self.conv(x)
        return res_x

    def non_linear(self, x):
        res_x = self.last_act(x)
        if not self.dropout is None:
            res_x = self.drop(res_x)
        return res_x

    def forward(self, x):
        x = self.get_features(x)
        x = self.non_linear(x)
        return x


class ds_res_convdown(nn.Module):
    '''
    res conv down with domain specific layers
    '''

    def __init__(self, in_ch, out_ch, num_domains=2, if_SN=False, bias=True, dropout=None):
        super(ds_res_convdown, self).__init__()
        # down-> conv3->prelu->conv
        if if_SN:
            self.down = spectral_norm(
                nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, bias=bias))

            self.conv_1 = spectral_norm(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias))
            self.norm_1 = DomainSpecificBatchNorm2d(
                out_ch, num_domains=num_domains)
            self.act_1 = nn.LeakyReLU(0.2)
            self.conv_2 = spectral_norm(nn.Conv2d(out_ch, out_ch,
                                                  3, padding=1, bias=bias))

            self.norm_2 = DomainSpecificBatchNorm2d(
                out_ch, num_domains=num_domains)
        else:
            self.down = nn.Conv2d(
                in_ch, in_ch, 3, stride=2, padding=1, bias=bias)

            self.conv_1 = spectral_norm(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias))
            self.norm_1 = DomainSpecificBatchNorm2d(
                out_ch, num_domains=num_domains)
            self.act_1 = nn.LeakyReLU(0.2)
            self.conv_2 = nn.Conv2d(out_ch, out_ch,
                                    3, padding=1, bias=bias)

            self.norm_2 = DomainSpecificBatchNorm2d(
                out_ch, num_domains=num_domains)

        if if_SN:
            self.conv_input = spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias))
        else:
            self.conv_input = nn.Conv2d(
                in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

        self.last_act = nn.LeakyReLU(0.2)
        self.dropout = dropout
        if not self.dropout is None:
            self.drop= Fixable2DDropout(p=dropout)
            # self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x, domain_id=0):
        x = self.down(x)
        f = self.conv_1(x)
        f = self.norm_1(f, domain_id)
        f = self.act_1(f)
        f = self.conv_2(f)
        f = self.norm_2(f, domain_id)
        res_x = self.last_act(self.conv_input(x) + f)
        if not self.dropout is None:
            res_x = self.drop(res_x)
        return res_x


class MyEncoder(nn.Module):
    '''
    Naive Encoder
    '''

    def __init__(self, input_channel, output_channel=None, feature_reduce=1, encoder_dropout=None, norm=nn.InstanceNorm2d, if_SN=False, act=torch.nn.Sigmoid()):
        super(MyEncoder, self).__init__()

        if if_SN:
            self.inc = nn.Sequential(
                spectral_norm(nn.Conv2d(input_channel, 64 // feature_reduce, 3, padding=1, bias=True)),
                norm(64 // feature_reduce),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64 // feature_reduce, 64 // feature_reduce, 3, padding=1, bias=True),
                norm(64 // feature_reduce),
            )
        else:
            self.inc = nn.Sequential(
                nn.Conv2d(input_channel, 64 // feature_reduce, 3, padding=1, bias=True),
                norm(64 // feature_reduce),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64 // feature_reduce, 64 // feature_reduce, 3, padding=1, bias=True),
                norm(64 // feature_reduce),
            )

        self.down1 = res_convdown(64 // feature_reduce, 128 // feature_reduce,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down2 = res_convdown(128 // feature_reduce, 256 // feature_reduce,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down3 = res_convdown(256 // feature_reduce, 512 // feature_reduce,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down4 = res_convdown(512 // feature_reduce, 512 // feature_reduce,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        if output_channel is None:
            self.final_conv = nn.Sequential(
                nn.Conv2d(512 // feature_reduce, 512 // feature_reduce, kernel_size=1, stride=1, padding=0),
                norm(512 // feature_reduce))
        else:
            self.final_conv = nn.Sequential(
                nn.Conv2d(512 // feature_reduce, output_channel, kernel_size=1, stride=1, padding=0),
                norm(output_channel))

        self.act = act
        for m in self._modules:
            normal_init(self._modules[m], 0, 0.02)

    def forward(self, x, domain_id=0):
        x1 = self.inc(x)
        x1 = F.leaky_relu(x1, negative_slope=0.2)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.final_conv(x5)
        self.before_act = x5
        if self.act is not None:
            x5 = self.act(x5)
        self.after_act = x5
        return x5


class DomainSpecificEncoder(nn.Module):
    '''
    Encoder with domain specific batch normalization layers.
    '''

    def __init__(self, input_channel, output_channel=None, num_domains=2, feature_reduce=1, encoder_dropout=None, if_SN=False, act=torch.nn.Sigmoid()):
        super(DomainSpecificEncoder, self).__init__()

        if if_SN:
            self.inc_conv_1 = spectral_norm(nn.Conv2d(input_channel, 64 //
                                                      feature_reduce, 3, padding=1, bias=True))
            self.norm_1 = DomainSpecificBatchNorm2d(
                64 // feature_reduce, num_domains)
            self.act_1 = nn.LeakyReLU(0.2)
            self.inc_conv_2 = spectral_norm(nn.Conv2d(64 // feature_reduce, 64 // feature_reduce,
                                                      3, padding=1, bias=True))
            self.norm_2 = DomainSpecificBatchNorm2d(
                64 // feature_reduce, num_domains)

        else:
            self.inc_conv_1 = nn.Conv2d(input_channel, 64 //
                                        feature_reduce, 3, padding=1, bias=True)
            self.norm_1 = DomainSpecificBatchNorm2d(
                64 // feature_reduce, num_domains)
            self.act_1 = nn.LeakyReLU(0.2)
            self.inc_conv_2 = nn.Conv2d(64 // feature_reduce, 64 // feature_reduce,
                                        3, padding=1, bias=True)
            self.norm_2 = DomainSpecificBatchNorm2d(
                64 // feature_reduce, num_domains)

        self.down1 = ds_res_convdown(
            64 // feature_reduce, 128 // feature_reduce, num_domains=num_domains, if_SN=if_SN, dropout=encoder_dropout)
        self.down2 = ds_res_convdown(
            128 // feature_reduce, 256 // feature_reduce, num_domains=num_domains, if_SN=if_SN, dropout=encoder_dropout)
        self.down3 = ds_res_convdown(
            256 // feature_reduce, 512 // feature_reduce, num_domains=num_domains, if_SN=if_SN, dropout=encoder_dropout)
        self.down4 = ds_res_convdown(
            512 // feature_reduce, 512 // feature_reduce, num_domains=num_domains, if_SN=if_SN, dropout=encoder_dropout)

        if output_channel is None:
            self.final_conv = nn.Conv2d(512 // feature_reduce, 512 // feature_reduce,
                                        kernel_size=1, stride=1, padding=0)
            self.final_norm = DomainSpecificBatchNorm2d(512 // feature_reduce, num_domains=num_domains)
        else:
            self.final_conv = nn.Conv2d(512 // feature_reduce, output_channel,
                                        kernel_size=1, stride=1, padding=0)
            self.final_norm = DomainSpecificBatchNorm2d(512 // feature_reduce, num_domains=num_domains)

        # apply sigmoid activation
        self.act = act  # torch.nn.LeakyReLU(0.2)

        for m in self._modules:
            normal_init(self._modules[m], 0, 0.02)

    def forward(self, x, domain_id=0):
        x1 = self.inc_conv_1(x)
        x1 = self.norm_1(x1, domain_id)
        x1 = self.act_1(x1)
        x1 = self.inc_conv_2(x1)
        x1 = self.norm_2(x1, domain_id)

        x1 = F.leaky_relu(x1, negative_slope=0.2)
        x2 = self.down1(x1, domain_id)
        x3 = self.down2(x2, domain_id)
        x4 = self.down3(x3, domain_id)
        x5 = self.down4(x4, domain_id)

        x5 = self.final_conv(x5)
        x5 = self.final_norm(x5, domain_id)

        if self.act is not None:
            x5 = self.act(x5)

        return x5


class MyDecoder(nn.Module):
    '''

    '''

    def __init__(self, input_channel, output_channel, feature_reduce=1, decoder_dropout=None, norm=nn.InstanceNorm2d, up_type='bilinear', if_SN=False, last_act=None):
        super(MyDecoder, self).__init__()
        self.up1 = res_up_family(input_channel, 256 // feature_reduce, norm=norm,
                                 up_type=up_type, dropout=decoder_dropout, if_SN=if_SN)
        self.up2 = res_up_family(256 // feature_reduce, 128 // feature_reduce, norm=norm,
                                 up_type=up_type, dropout=decoder_dropout, if_SN=if_SN)
        self.up3 = res_up_family(128 // feature_reduce, 64 // feature_reduce, norm=norm,
                                 up_type=up_type, dropout=decoder_dropout, if_SN=if_SN)
        self.up4 = res_up_family(64 // feature_reduce, 64 // feature_reduce, norm=norm,
                                 up_type=up_type, dropout=decoder_dropout, if_SN=if_SN)

        # final conv
        if if_SN:
            self.final_conv = spectral_norm(
                nn.Conv2d(64 // feature_reduce, output_channel, kernel_size=1, stride=1, padding=0))
        else:
            self.final_conv = nn.Conv2d(64 // feature_reduce, output_channel, kernel_size=1, stride=1, padding=0)
        self.last_act = last_act
        for m in self._modules:
            normal_init(self._modules[m], 0, 0.02)

    def forward(self, x):
        x2 = self.up1(x)
        x3 = self.up2(x2)
        x4 = self.up3(x3)
        x5 = self.up4(x4)
        self.hidden_feature= x5
        x5 = self.final_conv(x5)
        if self.last_act is not None:
            x5 = self.last_act(x5)
        return x5

    def apply_max_style(self, image_code,nn_style_augmentor_dict,decoder_layers_indexes=[3,4,5]):
        if 0 in decoder_layers_indexes:
            x = nn_style_augmentor_dict[str(0)](image_code.detach().clone())
        else:
            x = image_code.detach().clone()
        
        with _disable_tracking_bn_stats(self.up1):
            x2 = self.up1(x)
        if 1 in decoder_layers_indexes:
            x2 = nn_style_augmentor_dict[str(1)](x2)
        
        with _disable_tracking_bn_stats(self.up2):
            x3 = self.up2(x2)
        if 2 in decoder_layers_indexes:
            x3 = nn_style_augmentor_dict[str(2)](x3)
        
        with _disable_tracking_bn_stats(self.up3):
            x4 = self.up3(x3)
        if 3 in decoder_layers_indexes:
            x4 = nn_style_augmentor_dict[str(3)](x4)
        
        with _disable_tracking_bn_stats(self.up4):
            x5 = self.up4(x4)
        if 4 in decoder_layers_indexes:
            x5 = nn_style_augmentor_dict[str(4)](x5)
        
        with _disable_tracking_bn_stats(self.final_conv):
            x5 = self.final_conv(x5)
        if self.last_act is not None:
            x5 = self.last_act(x5)
        
        if 5 in decoder_layers_indexes:
            x5 = nn_style_augmentor_dict[str(5)](x5)
        return x5


class Dual_Branch_Encoder(nn.Module):
    '''
    FTN's encoder, produces two latent codes, z_i and z_s
    '''

    def __init__(self, input_channel, z_level_1_channel=None, z_level_2_channel=None, feature_reduce=1, encoder_dropout=None, norm=nn.InstanceNorm2d, if_SN=False, num_domains=1):
        super(Dual_Branch_Encoder, self).__init__()
        if num_domains > 1:
            self.general_encoder = DomainSpecificEncoder(input_channel, output_channel=z_level_1_channel, num_domains=num_domains,
                                                         feature_reduce=feature_reduce, encoder_dropout=encoder_dropout, if_SN=if_SN, act=torch.nn.ReLU())
        else:
            self.general_encoder = MyEncoder(input_channel, output_channel=z_level_1_channel,
                                             feature_reduce=feature_reduce, encoder_dropout=encoder_dropout, norm=norm, if_SN=if_SN, act=torch.nn.ReLU())

        if not if_SN:
            self.code_decoupler = nn.Sequential(
                nn.Conv2d(z_level_1_channel, z_level_2_channel, 3, padding=1, bias=False),
                norm(z_level_2_channel),
                nn.LeakyReLU(0.2),
                nn.Conv2d(z_level_2_channel, z_level_2_channel, 3, padding=1, bias=False),
                norm(z_level_2_channel),
                nn.ReLU(),


            )
        else:
            self.code_decoupler = nn.Sequential(
                spectral_norm(nn.Conv2d(z_level_1_channel, z_level_2_channel, 3, padding=1, bias=False)),
                norm(z_level_2_channel),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv2d(z_level_2_channel, z_level_2_channel, 3, padding=1, bias=False)),
                norm(z_level_2_channel),
                nn.ReLU(),

            )

        for m in self._modules:
            normal_init(self._modules[m], 0, 0.02)

    def filter_code(self, z):
        z_s = self.code_decoupler(z)
        return z_s

    def forward(self, x, domain_id=0):
        z_i = self.general_encoder(x, domain_id=domain_id)
        z_s = self.filter_code(z_i)
        return z_i, z_s


if __name__ == '__main__':

    encoder = DomainSpecificEncoder(input_channel=1, feature_reduce=4, if_SN=False, encoder_dropout=0.5)

    encoder.train()
    image = torch.autograd.Variable(torch.randn(2, 1, 192, 192))
    z1 = encoder(image, 1)
    print(z1.size())
