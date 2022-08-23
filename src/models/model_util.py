import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import contextlib
import math
import sys
sys.path.append('../')
# from medseg.models.segmentation_models.unet import UNet
from src.common_utils.basic_operations import check_dir
from src.models.custom_layers import BatchInstanceNorm2d,Fixable2DDropout,Fixable3DDropout


def get_normalization_params(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.InstanceNorm2d) or isinstance(module, BatchInstanceNorm2d):
            module.requires_grad_(True)
            module.train()
            module.track_running_stats = False
            for np, p in module.named_parameters():
                if np in ['weight', 'bias'] or p.requires_grad == True:
                    yield p


def get_conv_params(model, weight_only=False):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            for np, p in module.named_parameters():
                if np in ['weight', 'bias'] or p.requires_grad == True:
                    if weight_only:
                        if np in ['weight']:
                            yield p
                    else:
                        yield p


def replace_bn_with_in(model, name=None, affine=False, BN_IN=False):
    # replacing bn layers with in layers:
    # This function will recursively replace all relu module to selu module.
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # print('replaced: ', name, module.num_features)
            bn_weight = module.weight
            bn_bias = module.bias
            if BN_IN:
                new_module = BatchInstanceNorm2d(module.num_features, module.eps, momentum=module.momentum,
                                                 affine=affine, track_running_stats=affine)
            else:
                new_module = torch.nn.InstanceNorm2d(module.num_features, module.eps,
                                                     affine=affine, track_running_stats=affine)
            if affine:
                # print(bn_weight, bn_bias)
                new_module.weight = bn_weight
                new_module.bias = bn_bias
            setattr(model, name, new_module)
        else:
            replace_bn_with_in(module, name, affine, BN_IN)


def recover_model_w_bn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.InstanceNorm2d) or isinstance(module, BatchInstanceNorm2d):
            setattr(model, name, nn.BatchNorm2d(module.num_features, module.eps))
        else:
            recover_model_w_bn(module)

    # def replace_bn_with_in(m, name):
    #     for attr_str in dir(m):
    #         target_attr = getattr(m, attr_str)
    #         if type(target_attr) == torch.nn.BatchNorm2d:
    #             print('replaced: ', name, attr_str)
    #             setattr(m, attr_str, torch.nn.InstanceNorm2d(target_attr.num_features,
    #                                                          target_attr.eps))
    #     for n, ch in m.named_children():
    #         replace_bn_with_in(ch, n)

    # Partially based on: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone()
                                 for param in parameters
                                 if param.requires_grad]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        if len(self.collected_params) > 0:
            for c_param, param in zip(self.collected_params, parameters):
                if param.requires_grad:
                    param.data.copy_(c_param.data)
        else:
            print('did not find any copy, use the original params')


def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    if len(target.size()) == 3:
        target = target.view(target.numel())
        if not weight is None:
            # sum(weight) =C,  for numerical stability.
            weight = torch.softmax(weight, dim=0) * c
        loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
        if size_average:
            loss /= float(target.numel() + 1e-10)
    elif len(target.size()) == 4:
        # ce loss=-qlog(p)
        reference = F.softmax(target, dim=1)  # M,C
        reference = reference.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)  # M,C
        if weight is None:
            plogq = torch.mean(torch.mean(reference * log_p, dim=1))
        else:
            # sum(weight) =C
            weight = torch.softmax(weight, dim=0) * c
            plogq_class_wise = reference * log_p
            plogq_sum_class = 0.
            for i in range(plogq_class_wise.size(1)):
                plogq_sum_class += torch.mean(plogq_class_wise[:, i] * weight[i])
            plogq = plogq_sum_class
        loss = -1 * plogq
    else:
        raise NotImplementedError
    return loss


def clip_grad(optimizer):
    # https://github.com/rosinality/igebm-pytorch/blob/master/train.py
    # clip the gradient of parameters before optimization.
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))


def set_model_grad(model, state=False):
    assert model
    for p in model.parameters():  # reset requires_grad
        p.requires_grad = state


def set_grad(module, requires_grad=False):
    for p in module.parameters():  # reset requires_grad
        p.requires_grad = requires_grad


def make_one_hot(y, num_classes=4):
    batch_size, h, w = y.size(0), y.size(1), y.size(2)
    flatten_y = y.view(batch_size * h * w, 1)
    y_onehot = torch.zeros(batch_size * h * w, num_classes, dtype=torch.float32, device=y.device)
    y_onehot.scatter_(1, flatten_y, 1)
    y_onehot = y_onehot.view(batch_size, h, w, num_classes)
    y_onehot = y_onehot.permute(0, 3, 1, 2)
    y_onehot.requires_grad = False
    return y_onehot


def mask_latent_code_channel_wise(latent_code, decoder_function, label, num_classes=2, percentile=1 / 3.0, random=False, loss_type='corr', if_detach=True, if_soft=False):
    """
    given a latent code return a perturbed code where top % channels are masked
    Args:
        latent_code (torch tensor): latent code, z_i or z_s
        decoder_function (nn.module): a specific decoder function, which maps the latent code to the output space/image space
        label (torch tensor): targeted output, e.g. image or segmentation label
        num_classes (int): number of segmentation classes (incl. background), only used when 'label' is a labelmap
        percentile (float, optional): percentile of masked codes. Defaults to 1/3.0.
        random (bool, optional): if set to true, then randomly draw a threshold from (0,percentile) to mask. Defaults to False.
        loss_type (str, optional): name of the loss function. Defaults to 'corr'.
        if_detach (bool, optional): if false, will directly apply masking to the original code. Defaults to True.
        if_soft (bool, optional): if true, perform soft masking instead of hard masking. Defaults to False.

    Returns:
        [type]: [description]
    """
    '''

    '''
    use_gpu = True if latent_code.device != torch.device('cpu') else False

    code = makeVariable(latent_code, use_gpu=use_gpu, type='float', requires_grad=True)

    feature_channels = code.size(1)
    num_images = code.size(0)
    if len(label.size()) < len(code.size()):
        gt_y = make_one_hot(label, num_classes)
    else:
        gt_y = label

    if loss_type == 'corr':
        # self-challenging algorithm uses the correlation/similarity loss
        loss = torch.mean(decoder_function(code) * gt_y)
    elif loss_type == 'l1':
        loss = torch.nn.L1Loss()(decoder_function(code), gt_y)
    elif loss_type == 'mse' or loss_type == 'l2':
        loss = torch.mean((decoder_function(code) - gt_y)**2)
    elif loss_type == 'ce':
        logit = decoder_function(code)
        loss = cross_entropy_2D(input=logit, target=label, weight=None, size_average=True)
        loss = torch.mean(loss)

    gradient = torch.autograd.grad(loss, [code])[0]
    gradient_channel_mean = torch.mean(gradient.view(num_images, feature_channels, -1), dim=2)
    # select the threshold at top XX percentile
    # random percentile
    if random:
        percentile = np.random.rand() * percentile
    vector_thresh_percent = int(feature_channels * percentile)
    vector_thresh_value = torch.sort(gradient_channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]

    vector_thresh_value = vector_thresh_value.view(num_images, 1).expand(num_images, feature_channels)

    if if_soft:
        vector = torch.where(gradient_channel_mean > vector_thresh_value,
                             0.5 * torch.rand_like(gradient_channel_mean),
                             torch.ones_like(gradient_channel_mean))
    else:
        vector = torch.where(gradient_channel_mean > vector_thresh_value,
                             torch.zeros_like(gradient_channel_mean),
                             torch.ones_like(gradient_channel_mean))
    mask_all = vector.view(num_images, feature_channels, 1, 1)
    if not if_detach:
        masked_latent_code = latent_code * mask_all
    else:
        masked_latent_code = code * mask_all

    try:
        decoder_function.zero_grad()
    except:
        pass
    return masked_latent_code, mask_all


def mask_latent_code_spatial_wise(latent_code, decoder_function, label, num_classes, percentile=1 / 3.0, random=False,
                                  loss_type='corr', if_detach=True, if_soft=False):
    '''
    given a latent code return a perturbed code where top % areas are masked
    '''
    use_gpu = True if latent_code.device != torch.device('cpu') else False
    code = makeVariable(latent_code, use_gpu=use_gpu, type='float', requires_grad=True)
    num_images = code.size(0)
    spatial_size = code.size(2) * code.size(3)
    H, W = code.size(2), code.size(3)
    if len(label.size()) < len(code.size()):
        gt_y = make_one_hot(label, num_classes)
    else:
        gt_y = label

    if loss_type == 'corr':
        loss = torch.mean(decoder_function(code) * gt_y)
    elif loss_type == 'mse' or loss_type == 'l2':
        loss = torch.mean((decoder_function(code) - gt_y)**2)
    elif loss_type == 'l1':
        loss = torch.nn.L1Loss()(decoder_function(code), gt_y)
    elif loss_type == 'ce':
        logit = decoder_function(code)
        loss = cross_entropy_2D(input=logit, target=label, weight=None, size_average=True)
        loss = torch.mean(loss)

    gradient = torch.autograd.grad(loss, [code])[0]
    # mask gradient with largest response:
    spatial_mean = torch.mean(gradient, dim=1, keepdim=True)
    spatial_mean = spatial_mean.squeeze().view(num_images, spatial_size)

    # select the threshold at top XX percentile
    if random:
        percentile = np.random.rand() * percentile

    vector_thresh_percent = int(spatial_size * percentile)
    vector_thresh_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, vector_thresh_percent]

    vector_thresh_value = vector_thresh_value.view(num_images, 1).expand(num_images, spatial_size)

    if if_soft:
        vector = torch.where(spatial_mean > vector_thresh_value,
                             0.5 * torch.rand_like(spatial_mean),
                             torch.ones_like(spatial_mean))
    else:
        vector = torch.where(spatial_mean > vector_thresh_value,
                             torch.zeros_like(spatial_mean),
                             torch.ones_like(spatial_mean))

    mask_all = vector.view(num_images, 1, H, W)
    if not if_detach:
        masked_latent_code = latent_code * mask_all
    else:
        masked_latent_code = code * mask_all

    try:
        decoder_function.zero_grad()
    except:
        pass
    return masked_latent_code, mask_all


# def get_unet_model(model_path, num_classes=2, device=None, model_arch='UNet_16'):
#     '''
#     init model and load the trained parameters from the disk.
#     model path: string. path to the model checkpoint
#     device: torch device
#     return pytorch nn.module model
#     '''
#     assert check_dir(model_path) == 1, model_path + ' does not exists'
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     if model_arch == 'UNet_16':
#         model = UNet(input_channel=1, num_classes=num_classes, feature_scale=4)
#     elif model_arch == 'UNet_64':
#         model = UNet(input_channel=1, num_classes=num_classes, feature_scale=1)
#     else:
#         raise NotImplementedError
#     model.load_state_dict(torch.load(model_path))
#     model = model.to(device)
#     return model


def filter_unlabelled_predictions(predictions, threshold=0.8):
    '''
    given a batch of predictions,
    find the max prob for each pixel, if exceed the given threshhold return 1
    else return 0
    return: a batch of confidence maps NCHW, 0,1
    '''
    # find the maximum prob for each mask
    foreground_predictions = predictions.detach()
    max_prob_for_each_image = torch.max(foreground_predictions, dim=1)[0]
    max_prob_for_each_image = torch.clamp(max_prob_for_each_image - threshold, 0, 1)
    max_prob_for_each_image[foreground_predictions > 0] = 1
    confidence_maps = max_prob_for_each_image.unsqueeze(1).expand_as(predictions)
    return confidence_maps


def sharpen_predictions(predictions, temperature=0.5):
    '''
    shapen the predictions
    predictions: N*C*H*W: probabistic predictions (in mixmatch, this is an averaged value)
    '''
    predictions = F.softmax(predictions, dim=1)
    calibrated_p = predictions**(1 / temperature)
    return calibrated_p / calibrated_p.sum(axis=1, keepdims=True)


def stash_grad(model, grad_dict):
    for k, v in model.named_parameters():
        if v.grad is not None:
            if k in grad_dict.keys():
                grad_dict[k] += v.grad.clone()
            else:
                grad_dict[k] = v.grad.clone()
            model.zero_grad()
            # print ('gradient stashed')

    return grad_dict


def restore_grad(model, grad_dict):
    for k, v in model.named_parameters():
        if k in grad_dict.keys():
            grad = grad_dict[k]

            if v.grad is None:
                v.grad = grad
            else:
                v.grad += grad
    # print ('gradient restored')


def unit_norm(x, use_p_norm=False):
    # ## rescale
    abs_max = torch.max(
        torch.abs(x.view(x.size(0), -1)), 1, keepdim=True)[0].view(
        x.size(0), 1, 1, 1)

    x /= 1e-10 + abs_max
    # ## normalize
    if use_p_norm:
        batch_size = x.size(0)
        old_size = x.size()
        x = x.view(batch_size, -1)
        x = F.normalize(x, p=2, dim=1)
        x = x.view(old_size)

    return x


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    # turn off population stats in bn by setting eval =True instead of setting tracking_bn_stats=False due to the bug reported in https://github.com/pytorch/pytorch/issues/37823
    def switch_attr(model, new_state=None, hist_states=None):
        """[summary]

        Args:
            model ([torch.nn.Module]): [description]
            new_state ([bool], optional): [description]. Defaults to None.
            hist_states ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        old_states = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm3d):
                # print('here batch norm')
                old_states[name] = module.track_running_stats
                # old_state = module.track_running_stats
                if hist_states is not None:
                    module.track_running_stats = hist_states[name]
                    # module.train(hist_states[name])
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(hist_states[name])
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(hist_states[name])
                else:
                    if new_state is not None:
                        module.track_running_stats = new_state
                        # module.train(new_state)
                        if hasattr(module, 'weight'):
                            module.weight.requires_grad_(new_state)
                        if hasattr(module, 'bias'):
                            module.bias.requires_grad_(new_state)
            if isinstance(module, Fixable2DDropout) or isinstance(module, Fixable3DDropout):
                old_state = module.lazy_load ## freeze dropout to make the computation graph static
                module.lazy_load = not old_state
        return old_states

    old_states = switch_attr(model, False)
    yield
    switch_attr(model, hist_states=old_states)


class SizeEstimator(object):

    def __init__(self, model, input_size=(1, 1, 32, 32), bits=32):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.model = model
        self.input_size = input_size
        self.bits = 32
        return

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `models`'''
        mods = list(self.model.modules())
        sizes = []

        for i in range(1, len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        self.param_sizes = sizes
        return

    def get_output_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        input_ = Variable(torch.FloatTensor(*self.input_size), volatile=True)
        mods = list(self.model.modules())
        out_sizes = []
        for i in range(1, len(mods)):
            m = mods[i]
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

        self.out_sizes = out_sizes
        return

    def calc_param_bits(self):
        '''Calculate total number of bits to store `models` parameters'''
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def calc_forward_backward_bits(self):
        '''Calculate bits to store forward and backward pass'''
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = (total_bits * 2)
        return

    def calc_input_bits(self):
        '''Calculate bits to store input'''
        self.input_bits = np.prod(np.array(self.input_size)) * self.bits
        return

    def estimate_size(self):
        '''Estimate models size in memory in megabytes and bits'''
        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits + self.input_bits

        total_megabytes = (total / 8) / (1024 ** 2)
        return total_megabytes, total


def save_model_to_file(model_name, model, epoch, optimizer, save_path):
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    state = {'model_name': model_name,
             'epoch': epoch + 1,
             'model_state': state_dict,
             'optimizer_state': optimizer.state_dict()
             }
    torch.save(state, save_path)


def encode_3D(label_map, n_classes, use_gpu=False):
    '''
    input label as tensor
    return onehot label N*D*H*W
    :param label: batch_size*target_z*target_h*target_w
    :return:label:batch_size*n_classes*target_z*target_h*target_w
    '''
    # create one-hot vector for label map
    label_map = label_map[:, None, :, :, :]
    size = label_map.size()
    # print (size)
    oneHot_size = (size[0], n_classes, size[2], size[3], size[4])
    input_label = torch.zeros(torch.Size(oneHot_size)).float()
    if use_gpu:
        input_label = input_label.cuda()
        input_label = input_label.scatter_(1, label_map.long().cuda(), 1.0)
    else:
        input_label = input_label
        input_label = input_label.scatter_(1, label_map.long(), 1.0)

    return input_label


def encode_2D(label_map, n_classes, use_gpu=False):
    '''
    input label as tensor N*H*W
    return onehot label N*C*H*W
    :return:label:batch_size*n_classes*target_z*target_h*target_w
    '''
    # create one-hot vector for label map
    size = label_map[:, None, :, :].size()
    oneHot_size = (size[0], n_classes, size[2], size[3])
    input_label = torch.zeros(torch.Size(oneHot_size)).float()
    if use_gpu:
        input_label = input_label.cuda()
        input_label = input_label.scatter_(1, label_map[:, None, :, :].long().cuda(), 1.0)
    else:
        input_label = input_label
        input_label = input_label.scatter_(1, label_map[:, None, :, :].long(), 1.0)

    return input_label


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, initial_learning_rate, total_steps, power=0.985):
    lr = lr_poly(initial_learning_rate, i_iter, total_steps, power)
    print('lr', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def makeVariable(tensor, use_gpu=True, type='long', requires_grad=True):
    # conver type
    tensor = tensor.data
    if type == 'long':
        tensor = tensor.long()
    elif type == 'float':
        tensor = tensor.float()
    else:
        raise NotImplementedError

    # make is as Variable
    if use_gpu:
        variable = Variable(tensor.cuda(), requires_grad=requires_grad)
    else:
        variable = Variable(tensor, requires_grad=requires_grad)
    return variable


def get_scheduler(optimizer, lr_policy, lr_decay_iters=5, epoch_count=None, niter=None, niter_decay=None):
    print('lr_policy = [{}]'.format(lr_policy))
    if lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + epoch_count - niter) / float(niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.5)
    elif lr_policy == 'step2':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        print('schedular=plateau')
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=0.01, patience=5)
    elif lr_policy == 'plateau2':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'step_warmstart':
        def lambda_rule(epoch):
            # print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 100:
                lr_l = 1
            elif 100 <= epoch < 200:
                lr_l = 0.1
            elif 200 <= epoch:
                lr_l = 0.01
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step_warmstart2':
        def lambda_rule(epoch):
            # print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 50:
                lr_l = 1
            elif 50 <= epoch < 100:
                lr_l = 0.1
            elif 100 <= epoch:
                lr_l = 0.01
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:

        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)):
                self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale:
            self.rescale_output_array(x.size())

        return self.inputs, self.outputs
