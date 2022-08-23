'''

Robust and Generalizable Visual Representation
Learning via Random Convolutions
https://arxiv.org/pdf/2007.13003.pdf
'''

import random
import torch
import numpy as np


class RandConvAug():
    def __init__(self, kernel_size_candidates=[1, 3, 5, 7], prob=0.5, mix=True):
        self.kernel_size_candidates = kernel_size_candidates
        self.prob = prob
        self.mix = mix

    def transform(self, input_image):
        p0 = np.random.rand()
        if p0 < self.prob and not self.mix:
            return input_image
        else:
            # initialize kernel weights;
            ch = input_image.size(1)
            h = input_image.size(2)
            # random select one kernel size
            random.shuffle(self.kernel_size_candidates)
            k = self.kernel_size_candidates[0]
            stride = 1
            padding = ((h - 1) * stride - h + k) // 2
            # initialize weights w/ random gaussian dist N(0, 1/(c*k**2))
            sigma = 1 / torch.sqrt(torch.tensor(float(ch * (k**2))))
            weight = torch.randn(ch, ch, k, k, device=input_image.device) * (sigma)
            conv_filter = torch.nn.Conv2d(ch, ch, k, stride=stride, padding=padding, bias=False)
            conv_filter.to(input_image.device)
            conv_filter.weight = torch.nn.Parameter(weight)
            conv_filter.weight.requires_grad = False
            conv_image = conv_filter(input_image)

            if self.mix:
                alpha = torch.rand(1, device=input_image.device)
                mixed_image = alpha * (input_image) + (1 - alpha) * (conv_image)
            else:
                mixed_image = conv_image
            output_image = mixed_image.detach().clone()

        return output_image
