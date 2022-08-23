import torchvision
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import sys
sys.path.append('../')
from src.common_utils.morphology import Dilation2d


def basic_loss_fn(pred, target, loss_type='cross_entropy', class_weights=None, use_gpu=True):
    'this function contains basic segmentation losses in the supervised setting'
    num_classes = pred.size(1)
    if class_weights is None:
        class_weights = num_classes * [1. / num_classes]
    else:
        assert len(class_weights) == num_classes, 'each cls must have a weight, expect to have {} classses but got {} weights'.format(
            num_classes, len(class_weights))
    cls_weight_tensor = torch.tensor(class_weights, dtype=pred.dtype, device=pred.device)

    if loss_type == 'cross entropy':
        l = cross_entropy_2D(pred, target)
    elif loss_type == 'weighted cross entropy':
        l = cross_entropy_2D(pred, target, cls_weight_tensor)
    elif loss_type == 'dice':
        dice_func = SoftDiceLoss(n_classes=num_classes, use_gpu=use_gpu)
        l = dice_func(pred, target)
    elif loss_type == 'weighted dice':
        dice_func = SoftDiceLoss(n_classes=num_classes, use_gpu=use_gpu)
        l = dice_func(pred, target, weight=cls_weight_tensor)
    elif loss_type == 'foreground dice':
        cls_ids = [cid for cid in range(1, num_classes)]
        dice_func = SelectiveSoftDiceLoss(n_classes=num_classes, use_gpu=use_gpu, class_ids=cls_ids)
        l = dice_func(pred, target)
    elif loss_type == 'focal':
        loss_func = FocalLoss(gamma=2.)
        l = loss_func(pred, target)
    elif loss_type == 'contour_smooth':
        soft_max = torch.nn.Softmax2d()(pred)
        l = contour_loss(soft_max, target, num_classes=num_classes, use_gpu=use_gpu)
    else:
        raise NotImplementedError
    return l


def calc_angular_loss(tensor_4d, tensor_4d_reference):
    assert len(tensor_4d.size()) == 4, print(tensor_4d)
    assert len(tensor_4d_reference.size()) == 4, print(
        tensor_4d_reference.size())
    tensor_3d_a = tensor_4d.view(tensor_4d.size(
        0), tensor_4d.size(1), -1)  # NCHW->NCF
    tensor_3d_b = tensor_4d_reference.view(tensor_4d_reference.size(
        0), tensor_4d_reference.size(1), -1)  # NCHW->NCF
    loss = torch.mean(1 - torch.nn.CosineSimilarity(dim=-1)
                      (tensor_3d_a, tensor_3d_b))
    return loss


def calc_correlation(tensor_4d_a, tensor_4d_b):
    '''
    encourage the similarity between two spatial vectors in 4D format NCHW
    '''
    assert len(tensor_4d_a.size()) == 4, print(tensor_4d_a.size())
    assert len(tensor_4d_b.size()) == 4, print(tensor_4d_b.size())
    tensor_3d_a = tensor_4d_a.view(tensor_4d_a.size(
        0), tensor_4d_a.size(1), -1)  # NCHW->NCF
    tensor_3d_b = tensor_4d_b.view(tensor_4d_b.size(
        0), tensor_4d_a.size(1), -1)  # NCHW->NCF
    loss = torch.mean(torch.nn.CosineSimilarity(
        dim=-1)(tensor_3d_a, tensor_3d_b))
    return loss


def normalized_cross_correlation(x, y, return_map, reduction='mean', eps=1e-8):
    """ N-dimensional normalized cross correlation (NCC)
    Args:
        x (~torch.Tensor): Input tensor.
        y (~torch.Tensor): Input tensor.
        return_map (bool): If True, also return the correlation map.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    Returns:
        ~torch.Tensor: Output scalar
        ~torch.Tensor: Output tensor
    """

    shape = x.shape
    b = shape[0]

    # reshape
    x = x.view(b, -1)
    y = y.view(b, -1)

    # mean
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # deviation
    x = x - x_mean
    y = y - y_mean

    dev_xy = torch.mul(x, y)
    dev_xx = torch.mul(x, x)
    dev_yy = torch.mul(y, y)

    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                    torch.sqrt(torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    ncc_map = ncc.view(b, *shape[1:])

    # reduce
    if reduction == 'mean':
        ncc = torch.mean(torch.sum(ncc, dim=1))
    elif reduction == 'sum':
        ncc = torch.sum(ncc)
    else:
        raise KeyError('unsupported reduction type: %s' % reduction)

    if not return_map:
        return ncc

    return ncc, ncc_map


def calc_triplet_loss(anchor, positive, negative, distance_func=calc_angular_loss, margin=1):
    '''
        calc triplet loss l = [D(anchor,p)-D(anchor,n)+m]+, where D is defined as an angular distance function,
        margin is usually set a value between D_ap and D_an.
        reference: https://arxiv.org/pdf/1812.06576.pdf
    '''

    loss = F.relu((distance_func(anchor, positive)) -
                  distance_func(anchor, negative) + margin)
    return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin: float):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


def gram_matrix_2D(y):
    '''
    give torch 4d tensor, calculate Gram Matrix, y*y
    :param y:
    :return:
    '''
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)  # NCHW-> NCF
    features_t = features.transpose(1, 2)  # NFC
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def gram_matrix_3D(y):
    '''
    give torch 5d tensor, calculate Gram Matrix
    :param y:
    :return:
    '''
    (b, ch, z, h, w) = y.size()
    features = y.view(b, ch, z * w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * z * h * w)
    return gram


def style_loss(source, target):
    gram_diff = gram_matrix_3D(source) - gram_matrix_3D(target)
    loss = torch.mean(torch.mul(gram_diff, gram_diff))
    return loss


def cross_entropy_3D(input, target, weight=None, size_average=True):
    """[cross entropy for 3D segmentation]

    Args:
        input ([torch tensor]): network output (logits) before softmax.
        target ([torch tensor]): labelmaps, where each value is [0,n_classes-1]
        weight ([type], optional): a manual rescaling weight given to each class. If given, has to be a Tensor of size C. Defaults to None.
        size_average (bool, optional): [if true, average the loss]. Defaults to True.

    Returns:
        [type]: [description]
    """
    n, c, s, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(
        2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= float(target.numel())
    return loss


class NGF_Loss(torch.nn.Module):
    def __init__(self, epsilon=None, size_average=True,
                 reduction='mean', use_gpu=True):
        super(NGF_Loss, self).__init__()

        self._name = "ngf"
        self.size_average = size_average
        self.reduction = reduction
        self._epsilon = epsilon
        self.use_gpu = use_gpu

    def forward(self, input_image, reference_image):
        reference_image = reference_image.detach()
        fixed_gradient = self.compute_gradient(reference_image, normalize=False)
        moving_gradient = self.compute_gradient(input_image, normalize=False)
        self.fixed_gradient = fixed_gradient
        self.moving_gradient = moving_gradient

        value_x, valuemap_x = normalized_cross_correlation(
            self.fixed_gradient[:, 0], self.moving_gradient[:, 0], return_map=True)
        value_y, valuemap_y = normalized_cross_correlation(
            self.fixed_gradient[:, 1], self.moving_gradient[:, 1], return_map=True)

        value = 0.5 * (value_x + value_y)
        self.ncc_map = torch.sqrt(0.5 * (valuemap_x**2 + valuemap_y**2))

        if self.size_average:
            if self.reduction == 'mean':
                value = torch.mean(value)
            elif self.reduction == 'sum':
                value = torch.sum(value)
            else:
                raise ValueError
        value = 1 - value
        # print('loss:', value.item())
        return value

    def get_gaussian_kernel(self, kernel_size=5, sigma=8, channels=3):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        # Use n_sd sigmas
        if kernel_size < 2 * int(3.5 * sigma) + 1:
            # odd size so padding results in correct output size
            kernel_size = 2 * int(3.5 * sigma) + 1

        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
            torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) /
            (2 * variance)
        )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        pad_size = kernel_size // 2
        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, dilation=1, stride=1, bias=False,
                                    padding=pad_size)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        if self.use_gpu:
            gaussian_filter = gaussian_filter.cuda()
        return gaussian_filter

    def compute_gradient(self, image, normalize=False):
        # Approximate the gradient by short differences
        n, c, h, w = image.size()

        # smoothing the image
        image = self.get_gaussian_kernel(kernel_size=3, sigma=1, channels=image.size(1))(image)
        # use sobel filter to get dx and dy
        x_filter = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]]).reshape(1, 1, 3, 3)

        x_filter = np.repeat(x_filter, axis=1, repeats=image.size(1))
        x_filter = np.repeat(x_filter, axis=0, repeats=image.size(1))
        y_filter = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]]).reshape(1, 1, 3, 3)

        y_filter = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]]).reshape(1, 1, 3, 3)
        y_filter = np.repeat(y_filter, axis=1, repeats=image.size(1))
        y_filter = np.repeat(y_filter, axis=0, repeats=image.size(1))

        conv_x = nn.Conv2d(in_channels=image.size(1), out_channels=image.size(1), kernel_size=3, stride=1, padding=1,
                           dilation=1, bias=False)
        conv_x.weight = nn.Parameter(torch.from_numpy(x_filter).float())

        conv_y = nn.Conv2d(in_channels=image.size(1), out_channels=image.size(1), kernel_size=3, stride=1, padding=1,
                           dilation=1, bias=False)
        conv_y.weight = nn.Parameter(torch.from_numpy(y_filter).float())
        conv_y = conv_y.to(image.device)
        conv_x = conv_x.to(image.device)
        for param in conv_y.parameters():
            param.requires_grad = False
        for param in conv_x.parameters():
            param.requires_grad = False

        dx = conv_x(image)
        dy = conv_y(image)

        if normalize:
            # normalize gradient by its mag image
            epsilon = torch.mean(torch.abs(dx) + torch.abs(dy))
            mag = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2) + epsilon**2)
            ndx = torch. div(dx, mag)
            ndy = torch.div(dy, mag)
            gradient_field = torch.cat((ndx, ndy), dim=1)
        else:
            gradient_field = torch.cat((dx, dy), dim=1)
        return gradient_field


class EntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits):
        p = F.softmax(logits, dim=1)
        elementwise_entropy = -p * F.log_softmax(logits, dim=1)
        if self.reduction == 'none':
            return elementwise_entropy

        sum_entropy = torch.sum(elementwise_entropy, dim=1)
        if self.reduction == 'sum':
            return sum_entropy

        return torch.mean(sum_entropy)


class CrossEntropy2DLoss(nn.Module):
    def __init__(self,) -> None:
        super(CrossEntropy2DLoss, self).__init__()

    def forward(self, input, target, weight=None, size_average=True):
        loss = cross_entropy_2D(input, target, weight=None, size_average=size_average, mask=None, is_gt=False)
        return loss


def get_hierachical_loss(multi_preds, target, weights=[1., 1., 1.], use_gpu=True):
    '''
    cal loss with multiple level predictions
    make sure executing this after forwarding input
    :param target: the gt of the lv and myo and rv
    :return:
    '''
    assert len(multi_preds) == len(weights)
    binary_object_predict = multi_preds[0]
    binary_object_target = target.clone()
    binary_object_target[binary_object_target > 1] = 0

    object_loss = cross_entropy_2D(
        input=binary_object_predict, target=binary_object_target)

    biventricle_predict = multi_preds[1]
    biventricle_object_target = target.clone()
    # lv and myo =1
    biventricle_object_target[biventricle_object_target <= 2] = 1
    biventricle_object_target[biventricle_object_target > 2] = 2  # rv=2

    biventricle_loss = cross_entropy_2D(
        input=biventricle_predict, target=biventricle_object_target)

    # weighted loss which emphasizing the myocardium loss
    cls_weight_tensor = torch.tensor([0.2, 0.25, 0.3, 0.25]).float()
    if use_gpu:
        cls_weight_tensor = cls_weight_tensor.cuda()
    final_predict = multi_preds[2]

    final_predict_loss = cross_entropy_2D(
        input=final_predict, target=target, weight=cls_weight_tensor)

    loss = weights[0] * object_loss + weights[1] * \
        biventricle_loss + weights[2] * final_predict_loss

    return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target, weight=None, size_average=True):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if size_average:
            return loss.mean()
        else:
            return loss.sum()


#
class IntraSubjectLatentConsistencyLoss(nn.Module):
    '''
    latent space from different views should be similar
    '''

    def __init__(self):
        super(IntraSubjectLatentConsistencyLoss, self).__init__()

    def forward(self, x, weight=1):
        '''
        cal intra subject consistent loss
        :param x:  a list of latent space from k views [N*C*H'*W']_{1..k}
        :param weight:
        :return:
        '''
        flatten_z = []
        batch_size = x[0].size(0)

        for z in x:
            flatten_z.append(z.view(1, batch_size, -1))  # 1* N*F

        stack_z = torch.stack(flatten_z, dim=0)  # k*N*F
        averaged_z = torch.mean(stack_z, 0).unsqueeze(0)  # 1*N*F
        # print (averaged_z.size())
        bias = stack_z - averaged_z
        l2_loss = torch.mean(torch.mul(bias, bias))

        return l2_loss


class InterTemplateConsistencyLoss(nn.Module):
    def __init__(self):
        super(InterTemplateConsistencyLoss, self).__init__()

    def forward(self, x, weight=1):
        '''

        :param x: N*1*H*W reconstructed template before warping
        :param weight:
        :return:
        '''
        w = torch.cuda.FloatTensor(1).fill_(weight)
        w.cuda()
        w = Variable(w, requires_grad=False)
        averaged_template = torch.mean(x, 0).unsqueeze(0)  # 1*1*H*W
        bias = x - averaged_template
        l2_loss = w * torch.mean(torch.mul(bias, bias))
        self.loss = l2_loss
        return self.loss


def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def laplacian_smoothness_loss(input, target, num_classes, size_average=True, use_gpu=True):
    onehot_mapper = One_Hot(depth=num_classes, use_gpu=use_gpu)
    target = target.long()
    onehot_target = onehot_mapper(target).contiguous().view(
        input.size(0), num_classes, input.size(2), input.size(3))

    object_classes = num_classes - 1
    x_filter = np.array([[1, 0, 1],
                         [1, -8, 1],
                         [1, 0, 1]]).reshape(1, 1, 3, 3)
    x_filter = np.repeat(x_filter, axis=1, repeats=object_classes)
    # in channels
    x_filter = np.repeat(x_filter, axis=0, repeats=object_classes)
    #
    conv_x = nn.Conv2d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
                       dilation=1, bias=False)
    conv_x.weight = nn.Parameter(torch.from_numpy(x_filter).float())

    if use_gpu:
        conv_x = conv_x.cuda()
    for param in conv_x.parameters():
        param.requires_grad = False

    target_object_maps = onehot_target[:, 1:].float()
    input = input[:, 1:]

    g_x_pred = conv_x(input)
    g_x_truth = conv_x(target_object_maps)

    loss = torch.pow(torch.abs(g_x_pred - g_x_truth), 2)
    if size_average:
        return loss.mean()
    return loss


class SoftDiceLoss(nn.Module):

    # Dice loss: code is from https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/layers/loss
    # .py
    def __init__(self, n_classes, use_gpu=True, squared_union=False):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes, use_gpu).forward
        self.n_classes = n_classes
        self.squared_union = squared_union

    def forward(self, input, target, weight=None, mask=None, is_gt=False):
        smooth = 0.01
        batch_size = input.size(0)
        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        if len(target.size()) == 3:
            target = self.one_hot_encoder(target).contiguous().view(
                batch_size, self.n_classes, -1)
        elif len(target.size()) == 4 and target.size(1) == input.size(1):
            if not is_gt:
                target = F.softmax(target, dim=1).view(
                    batch_size, self.n_classes, -1)
            target = target.view(batch_size, self.n_classes, -1)
        else:
            print('the shapes for input and target do not match, input:{} target:{}'.format(
                str(input.size())), str(target.size()))
            raise ValueError
        if mask is not None:
            input = mask * input
            target = target * mask

        inter = torch.sum(input * target, 2) + smooth
        if self.squared_union:
            # 2pq/(|p|^2+|q|^2)
            union = torch.sum(input**2, 2) + torch.sum(target**2, 2) + smooth
        else:
            # 2pq/(|p|+|q|)
            union = torch.sum(input, 2) + torch.sum(target, 2) + smooth
        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score


class SoftDiceLoss3D(nn.Module):
    # Dice loss: code is from https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/layers/loss
    # .py
    def __init__(self, n_classes, class_ids, use_gpu=True, squared_union=False):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes, use_gpu).forward
        self.n_classes = n_classes
        self.class_ids = class_ids
        self.squared_union = squared_union

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)
        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        if len(target.size()) == 4 and len(input.size()) == 5:  # NDHW
            # convert it to one-hot maps
            target = self.one_hot_encoder(target).contiguous().view(
                batch_size, self.n_classes, -1)
        else:
            assert len(target.size()) == len(input.size())
        # only calc dice on selected classes.
        input = input[:, self.class_ids, :]
        target = target[:, self.class_ids, :]
        inter = torch.sum(input * target, 2) + smooth
        if self.squared_union:
            # 2pq/(|p|^2+|q|^2)
            union = torch.sum(input**2, 2) + torch.sum(target**2, 2) + smooth
        else:
            # 2pq/(|p|+|q|)
            union = torch.sum(input, 2) + torch.sum(target, 2) + smooth
        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))
        return score


class SelectiveSoftDiceLoss(nn.Module):
    def __init__(self, n_classes, use_gpu, class_ids, squared_union=False):
        super(SelectiveSoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes, use_gpu=use_gpu).forward
        self.n_classes = n_classes
        self.class_ids = class_ids
        self.squared_union = squared_union

    def forward(self, input, target, mask=None, is_gt=False):
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input, dim=1)

        if len(target.size()) == 3:
            target = self.one_hot_encoder(target).contiguous()
        else:
            if not is_gt:
                target = F.softmax(target, dim=1)
        if mask is not None:
            input = mask * input
            target = target * mask
        input = input[:, self.class_ids].view(
            batch_size, len(self.class_ids), -1)
        target = target.view(batch_size, self.n_classes, -1)
        target = target[:, self.class_ids, :]
        inter = torch.sum(input * target, 2)
        if self.squared_union:
            # 2pq/(|p|^2+|q|^2)
            union = torch.sum(input**2, 2) + torch.sum(target**2, 2)
        else:
            # 2pq/(|p|+|q|)
            union = torch.sum(input, 2) + torch.sum(target, 2)

        score = torch.sum((2.0 * inter + smooth) / (union + smooth))
        score = 1.0 - score / (float(batch_size) * float(len(self.class_ids)))
        # print ('score', score)
        return score


def entropy_loss(v, base=2, normalize=False, mask=None):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    if mask is None:
        mask = torch.ones(n, 1, h, w)
    if base == 2:
        emap = torch.mul(v, torch.log2(v + 1e-30))
        loss = -torch.sum(emap) / mask.sum()
        if normalize:
            loss = loss / np.log2(c)
    elif base == 'e':
        emap = torch.mul(v, torch.log(v + 1e-30))
        loss = -torch.sum(emap) / mask.sum()
        if normalize:
            loss = loss / np.log(c)
    elif base == '10':
        emap = torch.mul(v, torch.log(v + 1e-30))
        loss = -torch.sum(emap) / (n * h * w)
        if normalize:
            loss = loss / np.log10(c)
    return loss


class VGGPerceptualLoss(torch.nn.Module):
    # code is adapted from https://gist.github.com/brucemuller/37906a86526f53ec7f50af4e77d025c9
    def __init__(self, resize=True, layers=[1, 2, 3, 4], loss_fn=torch.nn.L1Loss(), use_gpu=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.layers = layers
        self.blocks = torch.nn.ModuleList(blocks)
        if use_gpu:
            self.blocks = self.blocks.cuda()
        self.transform = torch.nn.functional.interpolate
        if use_gpu:
            self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1))
            self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1))
        else:
            self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cpu').view(1, 3, 1, 1))
            self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cpu').view(1, 3, 1, 1))

        self.resize = resize
        self.loss_fn = loss_fn

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i + 1 in self.layers:
                l = self.loss_fn(x, y)
            #    print (l)
                loss = loss + l
        return loss


class One_Hot(nn.Module):
    def __init__(self, depth, use_gpu=True):
        super(One_Hot, self).__init__()
        self.depth = depth
        if use_gpu:
            self.ones = torch.sparse.torch.eye(depth).cuda()
        else:
            self.ones = torch.sparse.torch.eye(depth)

    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


class CustomBrierLoss(nn.Module):
    def __init__(self, n_classes, use_gpu=True):
        super(CustomBrierLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes, use_gpu).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        batch_size = input.size(0)
        # input
        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(target).contiguous().view(
            batch_size, self.n_classes, -1)
        # squared error between the one-hot encoding of the correct label and its associated probability
        l2_dist = torch.sum(torch.sum((input - target) ** 2, 2)) / \
            (float(batch_size) * float(self.n_classes))

        return l2_dist


def normalized_cross_correlation(x, y, return_map, reduction='mean', eps=1e-8):
    """ N-dimensional normalized cross correlation (NCC)
    Args:
        x (~torch.Tensor): Input tensor.
        y (~torch.Tensor): Input tensor.
        return_map (bool): If True, also return the correlation map.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    Returns:
        ~torch.Tensor: Output scalar
        ~torch.Tensor: Output tensor
    """

    shape = x.shape
    b = shape[0]

    # reshape
    x = x.view(b, -1)
    y = y.view(b, -1)

    # mean
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # deviation
    x = x - x_mean
    y = y - y_mean

    dev_xy = torch.mul(x, y)
    dev_xx = torch.mul(x, x)
    dev_yy = torch.mul(y, y)

    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                    torch.sqrt(torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    ncc_map = ncc.view(b, *shape[1:])

    # reduce
    if reduction == 'mean':
        ncc = torch.mean(torch.sum(ncc, dim=1))
    elif reduction == 'sum':
        ncc = torch.sum(ncc)
    else:
        raise KeyError('unsupported reduction type: %s' % reduction)

    if not return_map:
        return ncc

    return ncc, ncc_map


class CustomNormalizedCrossCorrelationLoss(nn.Module):
    '''
    Calc zero-normalized cross-correlation loss
    for applications when the brightness of the image and template can vary due to lighting and exposure conditions
    https://en.wikipedia.org/wiki/Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
    https://github.com/rogerberm/pytorch-ncc/blob/master/NCC.py
    Here image  and the template are of the same size.
    Therefore, their normalized cross-correlation equals the cosine of the angle between the unit vectors
    :cosine similarity between -1 (similar) and 1 (perfect match)
     loss: 1 ( dis) to -1 (perfect)
    '''

    def __init__(self, use_gpu=True, reduction='mean', zero_mean=True):
        super(CustomNormalizedCrossCorrelationLoss, self).__init__()
        self.use_gpu = use_gpu
        self.epsilon = 1e-6
        self.reduction = reduction
        self.zero_mean = zero_mean

    def forward(self, template, image):
        '''

        :param template: 4d tensor [bs, image_ch, h, w]
        :param image: 4d tensor [bs, image_ch, h, w]
        :return:

        '''

        return 1 - self.ncc(template, image)

    def ncc(self, template, image):
        batch_size = image.size(0)
        # substract template and image mean spatially
        if self.zero_mean:
            template_m = torch.mean(template, keepdim=True, dim=[2, 3])
            template = template - template_m
            image_m = torch.mean(image, keepdim=True, dim=[2, 3])
            image = image - image_m
        image_z_flatten = image.view(batch_size, -1)
        if template.size(0) == 1:
            # calc the cosine similarity between the two normalized vectors
            template = template.repeat(batch_size, 1, 1, 1)
        template_z_flatten = template.view(batch_size, -1)

        cos_func = nn.CosineSimilarity(dim=1, eps=self.epsilon)
        scores = cos_func(image_z_flatten, template_z_flatten)
        if self.reduction == 'mean':
            score = torch.sum(scores) / (1.0 * batch_size)
        elif self.reduction == 'sum':
            score = torch.sum(scores)
        elif self.reduction == 'none':
            score = scores
        else:
            raise NotImplementedError
        return score


class CustomLocalNormalizedCrossCorrelationLoss(nn.Module):
    ''''
    local (sliding window) normalized cross correlation (2D version)
    J. P. Lewis, "Fast Normalized Cross-Correlation", Industrial Light
           and Magic.
    '''

    def __init__(self, win_size=9, use_gpu=True, reduction='mean'):
        super(CustomLocalNormalizedCrossCorrelationLoss, self).__init__()
        self.use_gpu = use_gpu
        self.epsilon = 1e-6
        self.reduction = reduction
        self.win_size = win_size

    def get_conv_filter(self, image_channel, win_size):
        # gaussian kernels
        weights = torch.ones(image_channel, image_channel, win_size, win_size)
        pad_size = win_size // 2

        conv_kernel = nn.Conv2d(in_channels=image_channel, out_channels=image_channel, kernel_size=win_size, stride=1,
                                padding=pad_size,
                                bias=False)
        conv_kernel.weight = nn.Parameter(weights.float())

        for param in conv_kernel.parameters():
            param.requires_grad = False

        if self.use_gpu:
            conv_kernel.cuda()
        return conv_kernel

    def forward(self, template, image, mask=None):
        return 1 - self.ncc(template, image, mask)

    def ncc(self, template, image, mask=None):
        '''

        :param template: 4d tensor [bs, image_ch, h, w]
        :param image: 4d tensor [bs, image_ch, h, w]
        :return:
        a float loss

        '''
        batch_size = image.size(0)
        in_channel = image.size(1)
        h = image.size(2)
        w = image.size(3)
        ndims = len(image.size()) - 2
        assert ndims == 2, 'currently only support 2D input'
        assert self.win_size <= h and self.win_size <= w, 'window size must be smaller than image dimension'

        if not mask is None:
            template = template * mask
            image = image * mask

        I2 = template ** 2
        J2 = image ** 2
        IJ = image * template

        # compute local sums via convolution
        conv = self.get_conv_filter(
            image_channel=in_channel, win_size=self.win_size)

        I_sum = conv(template)
        J_sum = conv(image)
        I2_sum = conv(I2)
        J2_sum = conv(J2)
        IJ_sum = conv(IJ)
        # compute cross correlation
        win_area = self.win_size ** 2
        u_I = I_sum / (win_area * 1.0)
        u_J = J_sum / (win_area * 1.0)
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_area
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_area
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_area

        scores = cross / (torch.sqrt(I_var) * torch.sqrt(J_var) + self.epsilon)

        if self.reduction == 'mean':
            score = torch.mean(scores)
        elif self.reduction == 'sum':
            score = torch.sum(scores)
        elif self.reduction == 'none':
            score = scores
        else:
            raise NotImplementedError

        return score


class contrastive_loss(nn.Module):
    def __init__(self, tau=1, normalize=False):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj):

        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(
                1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(
                torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)
        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match /
                                     (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def cross_entropy_2D(input, target, weight=None, size_average=True, mask=None, is_gt=False):
    """[summary]
    calc cross entropy loss computed on 2D images
    Args:
        input ([torch tensor]): [4d logit] in the format of NCHW
        target ([torch tensor]): 3D labelmap or 4d logit (before softmax), in the format of NCHW
        weight ([type], optional): weights for classes. Defaults to None.
        size_average (bool, optional): take the average across the spatial domain. Defaults to True.
        mask : boolean mask, entries with 0 on the mask will be skipped when calc losses.
    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    if mask is None:
        mask = torch.ones(n, 1, h, w, device=log_p.device, requires_grad=False)
    mask = mask.reshape(n * h * w, 1)
    mask.requires_grad = False
    mask_region_size = float(torch.numel(mask))
    if not weight is None:
        weight = np.array(weight)
        weight = weight / (1.0 * sum(weight)) * c
        weight = torch.tensor(weight, device=input.device, dtype=torch.float32)
    if len(target.size()) == 3:
        target = target.view(target.numel())
        loss_vector = F.nll_loss(
            log_p, target, weight=weight, reduction="none")
        # print(loss_vector.size())
        loss_vector = loss_vector * mask.flatten()
        loss = torch.sum(loss_vector)
        if size_average:
            loss /= float(mask_region_size)  # /N*H'*W'
    elif len(target.size()) == 4:
        # ce loss=-qlog(p)
        if not is_gt:
            reference = F.softmax(target, dim=1)  # M,C
        else:
            reference = target
        reference = reference.transpose(1, 2).transpose(
            2, 3).contiguous().view(-1, c)  # M,C
        mask = mask.expand(n * h * w, c)
        if weight is None:
            plogq = torch.sum(reference * log_p * mask, dim=1)
            plogq = torch.sum(plogq)
            if size_average:
                plogq /= float(mask_region_size)
        else:
            plogq_class_wise = reference * log_p * mask
            plogq_sum_class = 0.
            for i in range(c):
                plogq_sum_class += torch.sum(plogq_class_wise[:, i] * weight[i])
            plogq = plogq_sum_class
            if size_average:
                # only average loss on the mask entries with value =1
                plogq /= float(mask_region_size)
        loss = -1 * plogq
    else:
        raise NotImplementedError
    return loss


def calc_segmentation_mse_consistency(input, target):
    loss = calc_segmentation_consistency(output=input, reference=target, divergence_types=[
        'mse'], divergence_weights=[1.0], class_weights=None, mask=None)
    return loss


def calc_segmentation_kl_consistency(input, target):
    loss = calc_segmentation_consistency(output=input, reference=target, divergence_types=[
        'kl'], divergence_weights=[1.0], class_weights=None, mask=None)
    return loss


def contour_loss(input, target, size_average=True, use_gpu=True, ignore_background=True, one_hot_target=True, mask=None, return_map=False):
    '''
    calc the contour loss across object boundaries (WITHOUT background class)
    :param input: NDArray. N*num_classes*H*W : pixelwise probs. for each class e.g. the softmax output from a neural network
    :param target: ground truth labels (NHW) or one-hot ground truth maps N*C*H*W
    :param size_average: batch mean
    :param use_gpu:boolean. default: True, use GPU.
    :param ignore_background:boolean, ignore the background class. default: True
    :param one_hot_target: boolean. if true, will first convert the target from NHW to NCHW. Default: True.
    :return:
    '''
    n, num_classes, h, w = input.size(0), input.size(
        1), input.size(2), input.size(3)
    if one_hot_target:
        onehot_mapper = One_Hot(depth=num_classes, use_gpu=use_gpu)
        target = target.long()
        onehot_target = onehot_mapper(target).contiguous().view(
            input.size(0), num_classes, input.size(2), input.size(3))
    else:
        onehot_target = target
    assert onehot_target.size() == input.size(), 'pred size: {} must match target size: {}'.format(
        str(input.size()), str(onehot_target.size()))

    if mask is None:
        # apply masks so that only gradients on certain regions will be backpropagated.
        mask = torch.ones_like(input).long().to(input.device)
        mask.requires_grad = False
    else:
        pass
        # print ('mask applied')

    if ignore_background:
        object_classes = num_classes - 1
        target_object_maps = onehot_target[:, 1:].float()
        input = input[:, 1:]
    else:
        target_object_maps = onehot_target
        object_classes = num_classes

    x_filter = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]]).reshape(1, 1, 3, 3)

    x_filter = np.repeat(x_filter, axis=1, repeats=object_classes)
    x_filter = np.repeat(x_filter, axis=0, repeats=object_classes)
    conv_x = nn.Conv2d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
                       dilation=1, bias=False)

    conv_x.weight = nn.Parameter(torch.from_numpy(x_filter).float())

    y_filter = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]]).reshape(1, 1, 3, 3)
    y_filter = np.repeat(y_filter, axis=1, repeats=object_classes)
    y_filter = np.repeat(y_filter, axis=0, repeats=object_classes)
    conv_y = nn.Conv2d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
                       bias=False)
    conv_y.weight = nn.Parameter(torch.from_numpy(y_filter).float())

    conv_x = conv_x.to(input.device)
    conv_y = conv_y.to(input.device)
    for param in conv_y.parameters():
        param.requires_grad = False
    for param in conv_x.parameters():
        param.requires_grad = False

    g_x_pred = conv_x(input) * mask[:, :object_classes]
    g_y_pred = conv_y(input) * mask[:, :object_classes]
    g_y_truth = conv_y(target_object_maps) * mask[:, :object_classes]
    g_x_truth = conv_x(target_object_maps) * mask[:, :object_classes]

    # mse loss
    loss = torch.nn.MSELoss(reduction='mean')(input=g_x_pred, target=g_x_truth) + \
        torch.nn.MSELoss(reduction='mean')(input=g_y_pred, target=g_y_truth)
    loss = 0.5 * loss
    if return_map:
        return loss, g_x_pred, g_y_pred, g_x_truth, g_y_truth
    return loss


def kl_divergence(reference, pred, mask=None, is_gt=False):
    '''
    calc the kl div distance between two outputs p and q from a network/model: p(y1|x1).p(y2|x2).
    :param reference p: directly output from network using origin input without softmax
    :param output q: approximate output: directly output from network using perturbed input without softmax
    :param is_gt: is onehot maps
    :return: kl divergence: DKL(P||Q) = mean(\sum_1 \to C (p^c log (p^c|q^c)))

    '''
    q = pred

    if mask is None:
        mask = torch.ones_like(q, device=q.device)
        mask.requires_grad = False
    if not is_gt:
        p = F.softmax(reference, dim=1)
        log_p = F.log_softmax(reference, dim=1)
    else:
        p = torch.where(reference == 0.0, 1e-8, 1)
        log_p = torch.log(p)  # avoid NAN when log(0)

    cls_plogp = mask * (p * log_p)
    cls_plogq = mask * (p * F.log_softmax(q, dim=1))
    plogp = torch.sum(cls_plogp, dim=1, keepdim=True)
    plogq = torch.sum(cls_plogq, dim=1, keepdim=True)

    kl_loss = torch.mean(plogp - plogq)
    return kl_loss


def calc_js_divergece(pred, target):
    """
    compute js divergence between two predictions
    Args:
        pred ([4-d tensor]): logits of the network output 1
        target ([4-d tensor]]):  logits of the network output 2

    Returns:
        js divergence:loss
    """
    # compute JS divergence
    c = pred.size(1)
    predict_softmax = F.softmax(pred, dim=1)
    target_softmax = F.softmax(target, dim=1)

    mean_predict_softmax = 0.5 * (predict_softmax + target_softmax)

    predict_softmax_transpose = predict_softmax.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target_softmax_transpose = target_softmax.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)

    mean_predict_softmax_log = torch.clamp(mean_predict_softmax, 1e-8, 1).log()
    mean_predict_softmax_log_transpose = mean_predict_softmax_log.transpose(
        1, 2).transpose(2, 3).contiguous().view(-1, c)

    kl_divergence_1 = F.kl_div(mean_predict_softmax_log_transpose, predict_softmax_transpose, reduction='batchmean')
    kl_divergence_2 = F.kl_div(mean_predict_softmax_log_transpose, target_softmax_transpose, reduction='batchmean')
    loss = 0.5 * (kl_divergence_1 + kl_divergence_2)
    return loss


def calc_segmentation_consistency(output, reference, divergence_types=['kl', 'contour'],
                                  divergence_weights=[1.0, 0.5], class_weights=None, scales=[0],
                                  mask=None, is_gt=False):
    """
    measuring the difference between two predictions (network logits before softmax)
    Args:
        output (torch tensor 4d): network predicts: NCHW (after perturbation)
        reference (torch tensor 4d): network references: NCHW (before perturbation)
        divergence_types (list, string): specify loss types. Defaults to ['kl','contour'].
        divergence_weights (list, float): specify coefficients for each loss above. Defaults to [1.0,0.5].
        scales (list of int): specify a list of downsampling rates so that losses will be calculated on different scales. Defaults to [0].
        mask ([tensor], 0-1 onehotmap): [N*1*H*W]. No losses on the elements with mask=0. Defaults to None.
    Raises:
        NotImplementedError: when loss name is not in ['kl','mse','contour']
    Returns:
        loss (tensor float):
    """
    dist = 0.
    num_classes = reference.size(1)
    if mask is None:
        # apply masks so that only gradients on certain regions will be backpropagated.
        mask = torch.ones_like(output).float().to(reference.device)
        mask .requires_grad = False
    for scale in scales:
        if scale > 0:
            output_reference = torch.nn.AvgPool2d(2 ** scale)(reference)
            output_new = torch.nn.AvgPool2d(2 ** scale)(output)
        else:
            output_reference = reference
            output_new = output
        for divergence_type, d_weight in zip(divergence_types, divergence_weights):
            loss = 0.
            if divergence_type == 'kl':
                '''
                standard kl loss
                '''
                loss = kl_divergence(
                    pred=output_new, reference=output_reference, mask=mask, is_gt=is_gt)
            elif divergence_type == 'ce':
                loss = cross_entropy_2D(
                    input=output_new, target=output_reference, mask=mask, is_gt=is_gt)
            elif divergence_type == 'weighted ce':
                assert class_weights is not None, 'must assign class weights'
                loss = cross_entropy_2D(
                    input=output_new, target=output_reference, mask=mask, is_gt=is_gt, weight=class_weights)
            elif divergence_type == 'Dice':
                use_gpu = False if output_reference.device == torch.device(
                    'cpu') else True
                loss = SoftDiceLoss(n_classes=num_classes, use_gpu=use_gpu)(
                    input=output_new, target=output_reference, mask=mask, is_gt=is_gt)
            elif divergence_type == 'mse':
                n, h, w = output_new.size(
                    0), output_new.size(2), output_new.size(3)
                if not is_gt:
                    target_pred = torch.softmax(output_reference, dim=1)
                else:
                    target_pred = output_reference
                input_pred = torch.softmax(output_new, dim=1)
                loss = torch.nn.MSELoss(reduction='sum')(
                    target=target_pred * mask, input=input_pred * mask)
                loss = loss / (n * h * w)
            elif divergence_type == 'contour':  # contour-based loss
                if not is_gt:
                    target_pred = torch.softmax(output_reference, dim=1)
                else:
                    target_pred = output_reference
                input_pred = torch.softmax(output_new, dim=1)
                cnt = 0
                for i in range(1, num_classes):
                    cnt += 1
                    loss += contour_loss(input=input_pred[:, [i], ], target=(target_pred[:, [i]]), ignore_background=False, mask=mask,
                                         one_hot_target=False)
                if cnt > 0:
                    loss /= cnt

            else:
                raise NotImplementedError

            print('{}:{}'.format(divergence_type, loss.item()))

            dist += 2 ** scale * (d_weight * loss)
    return dist / (1.0 * len(scales))


if __name__ == '__main__':
    pdist = nn.PairwiseDistance(p=2)
    cosine_similarity = nn.CosineSimilarity(dim=1)
    input1 = torch.ones(100, 128)
    input2 = torch.ones(100, 128)
    output = cosine_similarity(input1, input2)

    print('distance:', torch.mean(output, dim=0))

    source_feature = torch.autograd.Variable(torch.ones((1, 1, 36, 36)) * 2 + 1)
    # source_feature[2,2,1:5,1:5]=0.5
    # source_feature[2,1,2:3,2:3]=0.5

    target_feature = torch.autograd.Variable(torch.ones(10, 1, 36, 36) * 3 + 5)
    # target_feature[:,:,1:5,1:5]=0.5
    # target_feature[:,:,2:3,2:3]=0.5

    #  l = CustomEdgeLoss(n_classes=4,use_gpu=False)(source_feature,target_feature)
    #  print (l)
    # print (source_feature)
   # CORAL_loss = CORAL(source_feature, target_feature)
    correlation_loss = CustomNormalizedCrossCorrelationLoss(reduction='mean', zero_mean=True)(source_feature,
                                                                                              target_feature)
    local_correlation_loss = CustomLocalNormalizedCrossCorrelationLoss(win_size=9, use_gpu=False, reduction='mean')(
        source_feature, target_feature)

    # gram_diff= gram_matrix_2D(source_feature)-gram_matrix_2D(target_feature)
    # loss = torch.mean(torch.mul(gram_diff, gram_diff))
    # print(source_feature.shape)
    # print (loss.data)
    #
    # print(CORAL_loss.item())
    print(correlation_loss.size())
    print(local_correlation_loss.size())
    print(correlation_loss)
    print(local_correlation_loss)

    # smooth l1 loss
