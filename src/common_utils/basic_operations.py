# Created by cc215 at 27/12/19
# Enter feature description here
# Enter scenario name here
# Enter steps here
from numpy.lib.function_base import copy
import os
import shutil
import random
import os

import torch
from torch._C import device

import torch
import random
import numpy as np
import SimpleITK as sitk
import logging
import pickle
import sys
sys.path.append('../')
from src.dataset_loader.dataset_utils import resample_by_spacing, normalize_minmax_data

def save_dict(mydict, file_path):
    f = open(file_path,"wb")
    pickle.dump(mydict,f)
    
def load_dict(file_path):
    with open(file_path,"rb") as f:
        data = pickle.load(f)
    return data


def set_seed(seed):
    if seed is not None:
        global SEED
        SEED = seed
        os.environ['PYTHONHASHSEED'] = str(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        g = torch.Generator()
        g.manual_seed(SEED)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms = True
    else:
        torch.backends.cudnn.benchmark = True


def check_dir(dir_path, create=False):
    '''
    check the existence of a dir, when create is True, will create the dir if it does not exist.
    dir_path: str.
    create: bool
    return:
    exists (1) or not (-1)
    '''
    if os.path.exists(dir_path):
        return 1
    else:
        if create:
            os.makedirs(dir_path)
        return -1


def delete_dir(dir_path, is_link):
    if check_dir(dir_path) == 1:
        if is_link:
            os.unlink(dir_path)
        else:
            os.removedirs(dir_path)
        print('{} removed'.format(dir_path))


def move_dir(source_dir, target_dir, delete_source=False):
    if check_dir(source_dir) == 1:
        try:
            shutil.move(source_dir, target_dir)
        except:
            shutil.move(source_dir, target_dir, copy_function=shutil.copytree)
        print('success')
        print(os.listdir(target_dir))
        if delete_source:
            is_link = os.path.islink(source_dir)
            delete_dir(source_dir, is_link)
            print('deleted source:'.format(source_dir))
    else:
        print('error source dir {} not found'.format(source_dir))


def link_dir(source_dir, target_dir, target_is_directory=False):
    os.symlink(source_dir, target_dir, target_is_directory=target_is_directory)


def set_grad(module, requires_grad=False):
    for p in module.parameters():  # reset requires_grad
        p.requires_grad = requires_grad


def shuffle_tensor(input_tensor, right_shift=1):
    """
    given a batch of tensors
    we shuffle them by right shifting the order of images along the first dimension (N)
    Args:
        input_tensor (4d torch.tensor): a batch of images, dim; NCHW
        right_shift (int, optional): [description]. Defaults to 1. when right_shift == batch_size, then it has no effect.
    """
    # shuffle images in a batch, such that the segmentations do not match anymore.
    batch_size = input_tensor.size(0)
    device = input_tensor.device
    idx = torch.arange(batch_size, device=device)
    right_shift = right_shift % batch_size
    if right_shift > 0:
        idx += right_shift
        idx[-right_shift:] = torch.arange(right_shift, device=device)
        shuffled_image_batch = input_tensor[idx]
        input_tensor = shuffled_image_batch
    else:
        print('no shuffle')
    return input_tensor


def construct_input(segmentation, image=None, num_classes=None, apply_softmax=True, temperature=2, is_labelmap=False, smooth_label=False, shuffle=False, use_gpu=True):
    """
    concat image and segmentation toghether to form an input to an external assessor
    Args:
        image ([4d float tensor]): a of batch of images N(Ch)HW, Ch is the image channel
        segmentation ([4d float tensor] or 3d label map): corresponding segmentation map NCHW or 3 one hotmap NHW
        shuffle (bool, optional): if true, it will shuffle the input image and segmentation before concat. Defaults to False.
    """
    assert (apply_softmax and is_labelmap) is False

    if not is_labelmap:
        batch_size, h, w = segmentation.size(0), segmentation.size(2), segmentation.size(3)
    else:
        batch_size, h, w = segmentation.size(0), segmentation.size(1), segmentation.size(2)

    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    if not is_labelmap:
        if apply_softmax:
            assert len(segmentation.size()) == 4
            segmentation = segmentation / temperature  # soft probs
            softmax_predict = torch.softmax(segmentation, dim=1)
            segmentation = softmax_predict
    else:
        # make onehot maps
        assert num_classes is not None, 'please specify num_classes'
        flatten_y = segmentation.view(batch_size * h * w, 1)

        y_onehot = torch.zeros(batch_size * h * w, num_classes, dtype=torch.float32, device=device)
        y_onehot.scatter_(1, flatten_y, 1)
        y_onehot = y_onehot.view(batch_size, h, w, num_classes)
        y_onehot = y_onehot.permute(0, 3, 1, 2)
        y_onehot.requires_grad = False

        if smooth_label:
            # add noise to labels
            alpha = torch.rand(1, device=device) * 0.1
            y_onehot = (1 - alpha) * y_onehot + alpha / (num_classes)
        segmentation = y_onehot

    if shuffle and image is not None:
        # shuffle images in a batch, such that the segmentations do not match anymore.
        image = shuffle_tensor(image)

    if image is not None:
        image = image.detach()
        tuple = torch.cat([segmentation, image], dim=1)
        return tuple
    else:

        return segmentation


def recover_image(image, h_s, w_s, origin_h, origin_w):
    '''
    recover 3D  image to the original shape (before croping)
    image: 3D numpy nd array
    '''
    assert len(image.shape) == 3
    recover_image = np.zeros((image.shape[0], origin_h, origin_w), dtype=image.dtype)
    h, w = image.shape[1], image.shape[2]
    recover_image[:, h_s:h_s + h, w_s:w_s + w] = image
    return recover_image


def crop_or_pad(image, crop_size, label=None):
    '''
    crop or pad 2D/3D image-label pairs such that they can be processed by the network
    The maximum size is the crop size
    '''
    assert 2 <= len(image.shape) <= 3, 2 <= len(label.shape) <= 3
    if len(image.shape) == 2:
        h, w = image.shape[0], image.shape[1]
        n = 1
    else:
        n, h, w = image.shape[0], image.shape[1], image.shape[2]
    new_h, new_w = crop_size[0], crop_size[1]
    if new_h == h and new_w == w:
        return image, label, 0, 0, h, w

    h_s = (h - new_h) // 2
    w_s = (w - new_w) // 2
    if h <= new_h:
        pad_result = np.zeros((n, new_h, image.shape[2]), dtype=image.dtype)
        pad_result[:, -h_s:-h_s + h] = image
        image = pad_result
        if label is not None:
            pad_result = np.zeros((n, new_h, label.shape[2]), dtype=label.dtype)
            pad_result[:, -h_s:-h_s + h] = label
            label = pad_result
        h = new_h
    if w <= new_w:
        pad_result = np.zeros((n, image.shape[1], new_w), dtype=image.dtype)
        pad_result[:, :, -w_s:-w_s + w] = image
        image = pad_result
        if label is not None:
            pad_result = np.zeros((n, image.shape[1], new_w), dtype=label.dtype)
            pad_result[:, :, -w_s:-w_s + w] = label
            label = pad_result
        w = new_w

    h_s = (h - new_h) // 2
    w_s = (w - new_w) // 2
    if len(image.shape) == 2:
        image = image[h_s:h_s + new_h, w_s:w_s + new_w]
        if label is not None:
            label = label[h_s:h_s + new_h, w_s:w_s + new_w]
    else:
        image = image[:, h_s:h_s + new_h, w_s:w_s + new_w]
        if label is not None:
            label = label[:, h_s:h_s + new_h, w_s:w_s + new_w]
    return image, label, h_s, w_s, h, w


def switch_kv_in_dict(mydict):
    switched_dict = {y: x for x, y in mydict.items()}
    return switched_dict


def unit_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def intensity_norm_fn_selector(intensity_norm_type):
    if intensity_norm_type == 'min_max':
        return rescale_intensity
    elif intensity_norm_type == 'z_score':
        return z_score_intensity
    else:
        raise ValueError


def rescale_intensity(data, new_min=0, new_max=1, eps=1e-20):
    '''
    rescale pytorch batch data
    :param data: N*1*H*W
    :return: data with intensity ranging from 0 to 1
    '''
    orig_size = data.size()

    if len(data.size()) >=4:
        bs = data.size(0)
        c = data.size(1)
    elif len(data.size()) ==3:
        bs,h, w = data.size(0), data.size(1), data.size(2)
        c = 1
    else: raise ValueError
    try:
        data = data.view(bs * c, -1)
    except:
        data = data.contiguous()
        data = data.view(bs * c, -1)
    old_max = torch.max(data, dim=1, keepdim=True).values
    old_min = torch.min(data, dim=1, keepdim=True).values
    new_data = (data - old_min) / (old_max - old_min + eps) * (new_max - new_min) + new_min
    new_data = new_data.view(orig_size)
    return new_data


def z_score_intensity(data):
    '''
    rescale pytorch batch data
    :param data: N*c*H*W
    :return: data with intensity with zero mean dnd 1 std.
    '''
    orig_size = data.size()
    if len(data.size()) >=4:
        bs = data.size(0)
        c = data.size(1)
    elif len(data.size()) ==3:
        bs,h, w = data.size(0), data.size(1), data.size(2)
        c = 1
    else: raise ValueError
    try:
        data = data.view(bs * c, -1)
    except:
        data = data.contiguous()
        data = data.view(bs * c, -1)
    std, mean = torch.std_mean(data, dim=1, keepdim=True)
    std[std<=0]=1
    data = (data-mean)/std
    try:
        data = data.view(orig_size)
    except:
        data = data.contiguous()
        data = data.view(orig_size)
    return data


def load_img_label_from_path(img_path, label_path=None, new_spacing=None, normalize=False,keep_z_spacing=True,z_score=False):
    '''
    given two strings of image and label path
    return a tuple of 'image' ndarray, 'label' ndarray and sitk image and label.
    '''
    sitkImage = sitk.ReadImage(img_path)
    sitkImage = sitk.Cast(sitkImage, sitk.sitkFloat32)

    if not label_path is None:
        sitkLabel = sitk.ReadImage(label_path)
    if new_spacing is not None:
        new_spacing=list(new_spacing)
        if keep_z_spacing is True or new_spacing[2]<=0:
            new_spacing[2] = list(sitkImage.GetSpacing())[2]
        sitkImage = resample_by_spacing(sitkImage, new_spacing=new_spacing,
                                            interpolator=sitk.sitkLinear, keep_z_spacing=keep_z_spacing)
        if not label_path is None:
            sitkLabel = resample_by_spacing(sitkLabel, new_spacing=new_spacing,
                                            interpolator=sitk.sitkNearestNeighbor, keep_z_spacing=keep_z_spacing)
    ndarray = sitk.GetArrayFromImage(sitkImage)
    if normalize:
        if not z_score:
            ndarray = normalize_minmax_data(ndarray)
        else:
            ndarray = normalize_minmax_data(ndarray)

    if not label_path is None:
        label_ndarray = sitk.GetArrayFromImage(sitkLabel)
    if not label_path is None:
        return ndarray, label_ndarray, sitkImage, sitkLabel
    else:
        return ndarray, sitkImage
