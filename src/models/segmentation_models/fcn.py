import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
sys.path.append('../../')
from src.models.init_weight import init_weights
from src.models.segmentation_models.unet_parts import conv2DBatchNormRelu


class FCN(nn.Module):
    # Wenjia Bai's FCN pytorch Implementation, for more details, please ref to
    # https://jcmr-online.biomedcentral.com/articles/10.1186/s12968-018-0471-x
    def __init__(self, feature_scale=1, num_classes=4, input_channel=1, decoder_dropout=None):
        '''

        :param feature_scale: int, decrease the filters numbers by a factor of {feature_scale}
        :param num_classes: int.
        :param input_channel: int, 1 for gray images, 3 for RGB images
        :param decoder_dropout: bool, if true, then applying dropout to the concatenated features in the decoder path.
        '''
        super(FCN, self).__init__()
        self.in_channels = input_channel
        self.feature_scale = feature_scale
        self.n_classes = num_classes

        filters = [64, 128, 256, 512, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling as feature extractor
        self.conv1_1 = conv2DBatchNormRelu(self.in_channels, filters[0], 3, 1, 1, bias=True)
        self.conv1_2 = conv2DBatchNormRelu(filters[0], filters[0], 3, 1, 1, bias=True)

        self.conv2_1 = conv2DBatchNormRelu(filters[0], filters[1], stride=2, k_size=3, padding=1)
        self.conv2_2 = conv2DBatchNormRelu(filters[1], filters[1], stride=1, k_size=3, padding=1)

        self.conv3_1 = conv2DBatchNormRelu(filters[1], filters[2], stride=2, k_size=3, padding=1)
        self.conv3_2 = conv2DBatchNormRelu(filters[2], filters[2], stride=1, k_size=3, padding=1)
        self.conv3_3 = conv2DBatchNormRelu(filters[2], filters[2], stride=1, k_size=3, padding=1)

        self.conv4_1 = conv2DBatchNormRelu(filters[2], filters[3], stride=2, k_size=3, padding=1)
        self.conv4_2 = conv2DBatchNormRelu(filters[3], filters[3], stride=1, k_size=3, padding=1)
        self.conv4_3 = conv2DBatchNormRelu(filters[3], filters[3], stride=1, k_size=3, padding=1)

        self.conv5_1 = conv2DBatchNormRelu(filters[3], filters[4], stride=2, k_size=3, padding=1)
        self.conv5_2 = conv2DBatchNormRelu(filters[4], filters[4], stride=1, k_size=3, padding=1)
        self.conv5_3 = conv2DBatchNormRelu(filters[4], filters[4], stride=1, k_size=3, padding=1)
        # shape N*512*w/16*h/16
        ##
        self.level_5_out = conv2DBatchNormRelu(filters[4], filters[0], 3, 1, 1)  # 64
        self.level_4_out = conv2DBatchNormRelu(filters[3], filters[0], 3, 1, 1)
        self.level_3_out = conv2DBatchNormRelu(filters[2], filters[0], 3, 1, 1)
        self.level_2_out = conv2DBatchNormRelu(filters[1], filters[0], 3, 1, 1)
        self.level_1_out = conv2DBatchNormRelu(filters[0], filters[0], 3, 1, 1)

        self.up_5 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # concat all branches for prediction
        self.aggregate_layers = conv2DBatchNormRelu(filters[0] * 5, 64, 1, 1, 0)
        self.conv_final = conv2DBatchNormRelu(64, 64, 1, 1, 0)
        self.outS = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        # initialise weights

        self.dropout = decoder_dropout

    def extract(self, input):
        x = self.conv1_1(input)
        l1 = self.conv1_2(x)

        x = self.conv2_1(l1)
        l2 = self.conv2_2(x)

        x = self.conv3_1(l2)
        x = self.conv3_2(x)
        l3 = self.conv3_3(x)

        x = self.conv4_1(l3)
        x = self.conv4_2(x)
        l4 = self.conv4_3(x)

        x = self.conv5_1(l4)
        x = self.conv5_2(x)
        l5 = self.conv5_3(x)

        return {'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4, 'l5': l5}

    def forward(self, inputA):
        '''

        :param inputA: the one as input
        :return:
        '''
        feature_dict_A = self.extract(inputA)

        l1_concat = self.level_1_out(feature_dict_A['l1'])
        l2_concat = self.up_2(self.level_2_out(feature_dict_A['l2']))
        l3_concat = self.up_3(self.level_3_out(feature_dict_A['l3']))
        l4_concat = self.up_4(self.level_4_out(feature_dict_A['l4']))
        l5_concat = self.up_5(self.level_5_out(feature_dict_A['l5']))

        multi_level_features = torch.cat((l1_concat, l2_concat, l3_concat, l4_concat, l5_concat), dim=1)
        aggregated_output = self.aggregate_layers(multi_level_features)
        if not self.dropout is None:
            aggregated_output = F.dropout2d(aggregated_output, p=self.dropout, training=self.training)
        aggregated_output = self.conv_final(aggregated_output)
        if not self.dropout is None:
            aggregated_output = F.dropout2d(aggregated_output, p=self.dropout, training=self.training)

        segmentation = self.outS(aggregated_output)

        return segmentation

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p

    def get_net_name(self):
        return 'FCN'

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1_1)
        b.append(self.conv1_2)
        b.append(self.conv2_1)
        b.append(self.conv2_2)
        b.append(self.conv3_1)
        b.append(self.conv3_2)
        b.append(self.conv3_3)
        b.append(self.conv4_1)
        b.append(self.conv4_2)
        b.append(self.conv4_3)
        b.append(self.conv5_1)
        b.append(self.conv5_2)
        b.append(self.conv5_3)
        b.append(self.conv5_3)

        b.append(self.level_5_out)
        b.append(self.level_4_out)
        b.append(self.level_3_out)
        b.append(self.level_2_out)
        b.append(self.level_1_out)
        b.append(self.conv_final)
        b.append(self.aggregate_layers)

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
        b.append(self.outS.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]

    def cal_num_conv_parameters(self):
        cnt = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        if 'weight' in name:
                            print(name, param.data)
                            param = param.view(-1, 1)
                            param.squeeze()
                            cnt += len(param.data)
        print(cnt)


if __name__ == '__main__':
    from torch.autograd import Variable

    inputA = torch.rand((2, 1, 128, 128))
    # inputB=inputB[:,:,1:,:]
    # inputA=inputA[:,:,:-1,:]

    inputA_va = Variable(inputA)

    net = FCN(feature_scale=1, num_classes=2)
    net.eval()
    pred_A = net.forward(inputA_va)
    print(pred_A.size())
    net.cal_num_conv_parameters()
