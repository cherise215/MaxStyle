import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class Morphology(nn.Module):
    '''
    code is adapted from https://github.com/lc82111/pytorch_morphological_dilation2d_erosion2d/blob/master/morphology.py
    Base class for morpholigical operators 
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None, requires_grad=True, use_gpu=True):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure. 
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size,
                                                   kernel_size).cuda(), requires_grad=requires_grad)
        else:
            self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size,
                                                   kernel_size), requires_grad=requires_grad)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)

        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))

        # erosion
        weight = self.weight.view(self.out_channels, -1)  # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)

        if self.type == 'erosion2d':
            x = weight - x  # (B, Cout, Cin*kH*kW, L)
        elif self.type == 'dilation2d':
            x = weight + x  # (B, Cout, Cin*kH*kW, L)
        else:
            raise ValueError

        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False)  # (B, Cout, L)
        else:
            x = torch.logsumexp(x * self.beta, dim=2, keepdim=False) / self.beta  # (B, Cout, L)

        if self.type == 'erosion2d':
            x = -1 * x

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)

        return x


class Dilation2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20, requires_grad=True, use_gpu=True):
        super(Dilation2d, self).__init__(in_channels, out_channels,
                                         kernel_size, soft_max, beta, 'dilation2d', requires_grad, use_gpu)


class Erosion2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20, requires_grad=True, use_gpu=True):
        super(Erosion2d, self).__init__(in_channels, out_channels,
                                        kernel_size, soft_max, beta, 'erosion2d', requires_grad, use_gpu)


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


if __name__ == '__main__':
    # test
    x = torch.randn(1, 1, 10, 10).to(torch.device('cuda'))
    x[:, :, 5, 5] = 1
    e = Dilation2d(1, 1, 3, soft_max=False)
    print(x[0, 0])

    y = e(x)
    print(y[0, 0])
