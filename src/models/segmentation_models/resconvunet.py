# Created by cc215 at 07/06/19
# same unet but use convolutional kernel to do downsampling and upsampling; plus residual connection after downsampling and upsampling
# Enter scenario name here
# Enter steps here


import math
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import sys
sys.path.append('../../')
from src.models.custom_layers import BatchInstanceNorm2d
from src.models.custom_layers import Self_Attn
from src.models.segmentation_models.unet_parts import *


class ResConvUNet(nn.Module):
    '''
    reference https://github.com/Nishanksingla/UNet-with-ResBlock/blob/master/UNet_with_ResBlock.png
    '''

    def __init__(self, input_channel, num_classes, feature_scale=1, encoder_dropout=None, decoder_dropout=None, norm=nn.BatchNorm2d, self_attention=False, if_SN=False):
        super(ResConvUNet, self).__init__()
        self.inc = res_conv(input_channel, 64 // feature_scale, norm=norm, dropout=encoder_dropout)
        self.down1 = res_convdown(64 // feature_scale, 128 // feature_scale,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down2 = res_convdown(128 // feature_scale, 256 // feature_scale,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down3 = res_convdown(256 // feature_scale, 512 // feature_scale,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down4 = res_convdown(512 // feature_scale, 512 // feature_scale,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.up1 = res_conv_up(512 // feature_scale, 512 // feature_scale, 256 //
                               feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up2 = res_conv_up(256 // feature_scale, 256 // feature_scale, 128 //
                               feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up3 = res_conv_up(128 // feature_scale, 128 // feature_scale, 64 //
                               feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up4 = res_conv_up(64 // feature_scale, 64 // feature_scale, 64 // feature_scale,
                               norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        if self_attention:
            self.self_atn = Self_Attn(512 // feature_scale, 'relu')
        self.self_attention = self_attention
        self.outc = outconv(64 // feature_scale, num_classes)
        self.n_classes = num_classes
        self.attention_map = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        self.hidden_feature = x5
        if self.self_attention:
            x5, w_out, attention = self.self_atn(x5)
            self.attention_map = attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x

    def predict(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.self_attention:
            x5, w_out, attention = self.self_atn(x5)
            self.attention_map = attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3,)
        x = self.up3(x, x2,)
        x = self.up4(x, x1,)
        x = self.outc(x)
        if self.self_attention:
            return x, w_out, attention

        return x

    def get_net_name(self):
        return 'res unet'

    def adaptive_bn(self, if_enable=False):
        if if_enable:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, BatchInstanceNorm2d):
                    module.train()
                    module.track_running_stats = True

    def init_bn(self):
        for name, module in self.named_modules():
           # print(name, module)
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, BatchInstanceNorm2d):
                print(module)
                module.running_mean.zero_()
                module.running_var.fill_(1)

    def print_bn(self):
        for name, module in self.named_modules():
            # print(name, module)
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, BatchInstanceNorm2d):
                print(module.running_mean)
                print(module.running_var)

    def fix_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                for k in module.parameters():  # fix all conv layers
                    k.requires_grad = False
            elif 'outc' in name:
                if isinstance(module, nn.Conv2d):
                    for k in module.parameters():  # except last layers
                        k.requires_grad = True
            else:
                for k in module.parameters():  # fix all conv layers
                    k.requires_grad = False

    def get_adapted_params(self):
        for name, module in self.named_modules():
            # if isinstance(module,nn.BatchNorm2d):
            #     for p in module.parameters():
            #         yield p
            # if 'outc' in name:
            #     if isinstance(module,nn.Conv2d):
            #        for p in module.parameters(): ##fix all conv layers
            #            yield p
            for k in module.parameters():  # fix all conv layers
                if k.requires_grad:
                    yield k

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.inc)
        b.append(self.down1)
        b.append(self.down2)
        b.append(self.down3)
        b.append(self.down4)
        b.append(self.up1)
        b.append(self.up2)
        b.append(self.up3)
        b.append(self.up4)
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.outc.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]

    def cal_num_conv_parameters(self):
        cnt = 0
        for module_name, module in self.named_modules():
            print(module_name)
        for module_name, module in self.named_modules():
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Conv2d):
                print(module_name)
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        if 'weight' in name:
                            print(name, param.data)
                            param = param.view(-1, 1)
                            param.squeeze()
                            cnt += len(param.data)
        print(cnt)


if __name__ == '__main__':
    model = ResConvUNet(input_channel=1, num_classes=4, encoder_dropout=None, decoder_dropout=None, feature_scale=4)
    model.train()
    image = torch.autograd.Variable(torch.randn(2, 1, 224, 224), volatile=True)
    result = model(image)
    print(result.size())
    model.cal_num_conv_parameters()
