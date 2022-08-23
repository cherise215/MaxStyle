import math
import sys

import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import sys
sys.path.append('../../')
from src.models.model_util import _disable_tracking_bn_stats
from src.models.segmentation_models.unet_parts import *




class UnetEncoder(nn.Module):
    def __init__(self, input_channel, reduce_factor=1, encoder_dropout=None, norm=nn.BatchNorm2d, if_SN=False,activation=nn.ReLU, enable_code_filter=False):
        super(UnetEncoder, self).__init__()
        self.inc = inconv(input_channel, 64 // reduce_factor, norm=norm, dropout=encoder_dropout)
        self.down1 = down(64 // reduce_factor, 128 // reduce_factor, norm=norm, if_SN=if_SN, dropout=encoder_dropout,activation=activation)
        self.down2 = down(128 // reduce_factor, 256 // reduce_factor, norm=norm, if_SN=if_SN, dropout=encoder_dropout,activation=activation)
        self.down3 = down(256 // reduce_factor, 512 // reduce_factor, norm=norm, if_SN=if_SN, dropout=encoder_dropout,activation=activation)
        self.down4 = down(512 // reduce_factor, 512 // reduce_factor, norm=norm, if_SN=if_SN, dropout=encoder_dropout,activation=activation)
        self.enable_code_filter = enable_code_filter
    
        if self.enable_code_filter:
            self.code_filter_1 = CodeFilter(64 // reduce_factor, 64// reduce_factor, norm=norm, if_SN=if_SN) 
            self.code_filter_2 = CodeFilter(128 // reduce_factor, 128// reduce_factor, norm=norm, if_SN=if_SN) 
            self.code_filter_3 = CodeFilter(256 // reduce_factor, 256// reduce_factor, norm=norm, if_SN=if_SN) 
            self.code_filter_4 = CodeFilter(512 // reduce_factor, 512// reduce_factor, norm=norm, if_SN=if_SN) 
            self.code_filter_5 = CodeFilter(512 // reduce_factor, 512// reduce_factor, norm=norm, if_SN=if_SN) 
        else:
            self.code_filter_1 = None
            self.code_filter_2 = None
            self.code_filter_3 = None
            self.code_filter_4 = None
            self.code_filter_5 = None
        
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
        return [x1, x2, x3, x4, x5]

    def filter_code(self, z):
        ##extract shape related code from given image code
        [z1, z2, z3, z4, z5]  =z
        if self.enable_code_filter:
            z1 = self.code_filter_1(z1)
            z2 = self.code_filter_2(z2)
            z3 = self.code_filter_3(z3)
            z4 = self.code_filter_4(z4)
            z5 = self.code_filter_5(z5)
        return [z1, z2, z3, z4, z5]
    


class UnetDecoder(nn.Module):
    def __init__(self, n_classes, reduce_factor=1, decoder_dropout=None,up_type='bilinear',  norm=nn.BatchNorm2d, activation=nn.ReLU,if_SN=False,last_act=None):
        super(UnetDecoder, self).__init__()
        self.up1 = up(512 // reduce_factor, 512 // reduce_factor, 256 //
                      reduce_factor, up_type=up_type,norm=norm, dropout=decoder_dropout, if_SN=if_SN,activation=activation)
        self.up2 = up(256 // reduce_factor, 256 // reduce_factor, 128 //
                      reduce_factor, up_type=up_type,norm=norm, dropout=decoder_dropout, if_SN=if_SN,activation=activation)
        self.up3 = up(128 // reduce_factor, 128 // reduce_factor, 64 //
                      reduce_factor, up_type=up_type,norm=norm, dropout=decoder_dropout, if_SN=if_SN,activation=activation)
        self.up4 = up(64 // reduce_factor, 64 // reduce_factor, 64 // reduce_factor,up_type=up_type,activation=activation,
                      norm=norm, dropout=decoder_dropout, if_SN=if_SN)

        self.outc = outconv(64 // reduce_factor, n_classes)
        self.n_classes = n_classes
        self.last_act = last_act

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
    def forward(self, cascaded_features):
        '''
        decode segmentation from cascaded features
        '''
        x1 = cascaded_features[0]
        x2 = cascaded_features[1]
        x3 = cascaded_features[2]
        x4 = cascaded_features[3]
        x5 = cascaded_features[4]
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if not self.last_act is None:
            x = self.last_act(x)
        return x

    def apply_max_style(self,cascaded_features, decoder_layers_indexes,nn_style_augmentor_dict):
        x1 = cascaded_features[0]
        x2 = cascaded_features[1]
        x3 = cascaded_features[2]
        x4 = cascaded_features[3]
        x5 = cascaded_features[4]
        if 0 in decoder_layers_indexes:
            x5 = nn_style_augmentor_dict[str(0)](x5.detach().clone())
        else:
            x5 = x5.detach().clone()
        with _disable_tracking_bn_stats(self.up1):
            x = self.up1(x5, x4)
        if 1 in decoder_layers_indexes:
            x = nn_style_augmentor_dict[str(1)](x)
        with _disable_tracking_bn_stats(self.up2):
            x = self.up2(x, x3)
        if 2 in decoder_layers_indexes:
            x = nn_style_augmentor_dict[str(2)](x)
        with _disable_tracking_bn_stats(self.up3):
            x = self.up3(x, x2)
        if 3 in decoder_layers_indexes:
            x = nn_style_augmentor_dict[str(3)](x)
        with _disable_tracking_bn_stats(self.up4):
            x = self.up4(x, x1)
        if 4 in decoder_layers_indexes:
            x = nn_style_augmentor_dict[str(4)](x)
        with _disable_tracking_bn_stats(self.outc):
            x = self.outc(x)
        if self.last_act is not None:
            x = self.last_act(x)
        if 5 in decoder_layers_indexes:
            x = nn_style_augmentor_dict[str(5)](x)
        return x

class UNet(nn.Module):
    def __init__(self, input_channel, num_classes, feature_scale=1, encoder_dropout=None, decoder_dropout=None, norm=nn.BatchNorm2d, if_SN=False, last_layer_act=None):
        super(UNet, self).__init__()
        self.inc = inconv(input_channel, 64 // feature_scale, norm=norm, dropout=encoder_dropout)
        self.down1 = down(64 // feature_scale, 128 // feature_scale, norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down2 = down(128 // feature_scale, 256 // feature_scale, norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down3 = down(256 // feature_scale, 512 // feature_scale, norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down4 = down(512 // feature_scale, 512 // feature_scale, norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.up1 = up(512 // feature_scale, 512 // feature_scale, 256 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up2 = up(256 // feature_scale, 256 // feature_scale, 128 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up3 = up(128 // feature_scale, 128 // feature_scale, 64 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up4 = up(64 // feature_scale, 64 // feature_scale, 64 // feature_scale,
                      norm=norm, dropout=decoder_dropout, if_SN=if_SN)

        self.outc = outconv(64 // feature_scale, num_classes)
        self.n_classes = num_classes
        self.last_act = last_layer_act

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        cascaded_features = self.encode_image(x)
        self.hidden_feature = cascaded_features[-1]
        x = self.decode_segmentation(cascaded_features)
        return x

    def predict(self, x):
        with torch.inference_mode():
            self.eval()
            y = self.forward(x)
            return y

    def encode_image(self,x):
        '''
        encode image to cascaded features
        '''
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]
    
    def decode_segmentation(self,cascaded_features):
        '''
        decode segmentation from cascaded features
        '''
        x1 = cascaded_features[0]
        x2 = cascaded_features[1]
        x3 = cascaded_features[2]
        x4 = cascaded_features[3]
        x5 = cascaded_features[4]
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if not self.last_act is None:
            x = self.last_act(x)
        return x

    def get_net_name(self):
        return 'unet'

    def adaptive_bn(self, if_enable=False):
        if if_enable:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train()
                    module.track_running_stats = True

    def init_bn(self):
        for name, module in self.named_modules():
           # print(name, module)
            if isinstance(module, nn.BatchNorm2d):
                print(module)
                module.running_mean.zero_()
                module.running_var.fill_(1)

    def fix_conv_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Conv2d):
                # print(name)
                for k in module.parameters():  # fix all conv layers
                    k.requires_grad = False

            else:
                for k in module.parameters():
                    k.requires_grad = True

    def activate_conv_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Conv2d):
                # print(name)
                for k in module.parameters():
                    k.requires_grad = True

    def print_bn(self):
        for name, module in self.named_modules():
            # print(name, module)
            if isinstance(module, nn.BatchNorm2d):
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


class DeeplySupervisedUNet(nn.Module):
    def __init__(self, input_channel, num_classes, base_n_filters=64, dropout=None, activation=nn.ReLU):
        super(DeeplySupervisedUNet, self).__init__()
        self.inc = inconv(input_channel, base_n_filters, activation=activation)
        self.down1 = down(base_n_filters, base_n_filters * 2, activation=activation)
        self.down2 = down(base_n_filters * 2, base_n_filters * 4, activation=activation)
        self.down3 = down(base_n_filters * 4, base_n_filters * 8, activation=activation)
        self.down4 = down(base_n_filters * 8, base_n_filters * 8, activation=activation)
        self.up1 = up(base_n_filters * 8, base_n_filters * 8, base_n_filters *
                      4, activation=activation, dropout=dropout)
        self.up2 = up(base_n_filters * 4, base_n_filters * 4, base_n_filters *
                      2, activation=activation, dropout=dropout)
        self.up3 = up(base_n_filters * 2, base_n_filters * 2, base_n_filters, activation=activation, dropout=dropout)
        self.up4 = up(base_n_filters, base_n_filters, base_n_filters, activation=activation)
        self.up2_conv1 = outconv_relu(base_n_filters * 2, num_classes, activation=None)
        self.up2_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3_conv1 = outconv_relu(base_n_filters, num_classes, activation=None)
        self.up3_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.outc = outconv(base_n_filters, num_classes)
        self.n_classes = num_classes
        self.dropout = dropout
        if dropout is not None:
            # self.dropoutlayer = nn.Dropout2d(p=dropout)
            self.dropoutlayer= Fixable2DDropout(p=dropout)

        else:
            self.dropoutlayer= Fixable2DDropout(p=0)
            # self.dropoutlayer = nn.Dropout2d(p=0)

    def forward(self, x, multi_out=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.dropoutlayer(self.down2(x2))  # tail it after pooling
        x4 = self.dropoutlayer(self.down3(x3))
        x5 = self.dropoutlayer(self.down4(x4))

        x = self.up1(x5, x4)
        x_2 = self.up2(x, x3)  # insert dropout after concat
        dsv_x_2 = self.up2_conv1(x_2)
        dsv_x_2_up = self.up2_up(dsv_x_2)

        x_3 = self.up3(x_2, x2)
        dsv_x_3 = self.up3_conv1(x_3)
        dsv_mixed = dsv_x_2_up + dsv_x_3
        dsv_mixed_up = self.up3_up(dsv_mixed)

        x_4 = self.up4(x_3, x1)
        out = self.outc(x_4)
        final_output = torch.add(out, dsv_mixed_up)
        if multi_out:
            return out, dsv_mixed_up, final_output

        return final_output

    def predict(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x_2 = self.up2(x, x3)
        dsv_x_2 = self.up2_conv1(x_2)
        dsv_x_2_up = self.up2_up(dsv_x_2)

        x_3 = self.up3(x_2, x2)
        dsv_x_3 = self.up3_conv1(x_3)
        dsv_mixed = dsv_x_2_up + dsv_x_3
        dsv_mixed_up = self.up3_up(dsv_mixed)

        x_4 = self.up4(x_3, x1)
        out = self.outc(x_4)
        final_output = torch.add(out, dsv_mixed_up)

        return final_output

    def get_net_name(self):
        return 'dsv_unet'

    def adaptive_bn(self, if_enable=False):
        if if_enable:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    # if 'down' in name or 'up' in name or 'inc' in name:
                    print(module.name)
                    module.train()
                    module.track_running_stats = True

    def init_bn(self):
        for name, module in self.named_modules():
            # print(name, module)
            if isinstance(module, nn.BatchNorm2d):
                print(module)
                module.running_mean.zero_()
                module.running_var.fill_(1)

    def fix_params(self):
        for name, param in self.named_parameters():
            if 'outc' in name:
                # initialize
                if 'conv' in name and 'weight' in name:
                    n = param.size(0) * param.size(2) * param.size(3)
                    param.data.normal_().mul_(math.sqrt(2. / n))
            else:
                param.requires_grad = False

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


class UNetv2(nn.Module):
    def __init__(self, input_channel, num_classes, feature_scale=1, encoder_dropout=None, decoder_dropout=None, norm=nn.BatchNorm2d, if_SN=False, last_layer_act=None):
        super(UNetv2, self).__init__()
        self.inc = inconv(input_channel, 64 // feature_scale, norm=norm, dropout=encoder_dropout)
        self.down1 = down(64 // feature_scale, 128 // feature_scale, norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down2 = down(128 // feature_scale, 256 // feature_scale, norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down3 = down(256 // feature_scale, 512 // feature_scale, norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down4 = down(512 // feature_scale, 1024 // feature_scale, norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.up1 = up(1024 // feature_scale, 512 // feature_scale, 256 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up2 = up(256 // feature_scale, 256 // feature_scale, 128 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up3 = up(128 // feature_scale, 128 // feature_scale, 64 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up4 = up(64 // feature_scale, 64 // feature_scale, 64 // feature_scale,
                      norm=norm, dropout=decoder_dropout, if_SN=if_SN)

        self.outc = outconv(64 // feature_scale, num_classes)
        self.n_classes = num_classes
        self.attention_map = None
        self.last_act = last_layer_act

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

        if not self.last_act is None:
            x = self.last_act(x)

        return x

    def predict(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3,)
        x = self.up3(x, x2,)
        x = self.up4(x, x1,)
        x = self.outc(x)

        return x

    def get_net_name(self):
        return 'unet'

    def adaptive_bn(self, if_enable=False):
        if if_enable:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train()
                    module.track_running_stats = True

    def init_bn(self):
        for name, module in self.named_modules():
           # print(name, module)
            if isinstance(module, nn.BatchNorm2d):
                print(module)
                module.running_mean.zero_()
                module.running_var.fill_(1)


if __name__ == '__main__':
    model = UNet(input_channel=1, feature_scale=1, num_classes=4, encoder_dropout=0.3)
    model.train()
    image = torch.autograd.Variable(torch.randn(2, 1, 224, 224))
    result = model(image)
    print(model.hidden_feature.size())
    print(result.size())

    # model.cal_num_conv_parameters()
    # model.adaptive_bn()
