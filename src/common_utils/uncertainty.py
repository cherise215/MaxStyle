# Created by cc215 at 22/05/19
# this file contains common method to estimate uncertainty
# using MC dropout
import numpy as np


def cal_entropy_maps(prediction_logit, eps=1e-7, threhold=0., use_max=False, temperature=1, normalize=False):
    '''

    calculate entropy maps from network logits which is the output from neural networks before the softmax layer.
     eps is used to prevent np.log2(zero). Note that this function is working on one image only.
    :param 3D logits: C_classes*H*W
    :param threshold: float uncertainty below this value will be filtered out.
    :param use_max: if use max, then find the maximum prob over classes and use the value to cal entropy,
    other wise calculate entropy across each channel and then averaged them.
    :return: A 2D map  H*W with values >0
    '''
    C = prediction_logit.shape[0]
    prediction_logit = np.array(prediction_logit)
    pred_probs = np.exp(prediction_logit / temperature) / \
        np.sum(np.exp(prediction_logit / temperature), keepdims=True, axis=0)
    assert len(pred_probs.shape) == 3, 'only support input of three dimension [Channel, H, W]'
    if use_max:
        ax_probs = np.amax(pred_probs, axis=0)  # using maxium prob to cal entropy
        entropy = (-ax_probs * np.log2(ax_probs + eps))
    else:
        # sum entropy
        entropy = (-pred_probs * np.log2(pred_probs + eps)).sum(axis=0)
    entropy = np.nan_to_num(entropy)
    if normalize:
        assert use_max is False
        # shannon entropy, entropy has been rescaled to  [0,1]
        entropy = entropy / np.log2(C)

    entropy[entropy < threhold] = 0.
    return entropy


def cal_batch_entropy_maps(pred_logits, eps=1e-7, threhold=0., use_max=False, temperature=1, normalize=True):
    '''

    calculate entropy maps from logits (numpy) which is the output from neural networks before the softmax layer.
     eps is used to prevent np.log2(zero). Note that this function is working on batches of image.
    :param 4D batch input (logits): N*C_classes*H*W
    :param threshold: float uncertainty below this value will be filtered out.
    :param use_max: if use max, then find the maximum prob over classes and use the value to cal entropy,
    other wise calculate entropy across each channel and then averaged them.
    :return: A 3D map [ N*H*W] with values >0
    '''

    assert len(pred_logits.shape) == 4, 'only support input of four dimension [N, Channel, H, W]'
    C = pred_logits.shape[1]
    prediction_logit = np.array(pred_logits)
    pred_probs = np.exp(prediction_logit / temperature) / \
        np.sum(np.exp(prediction_logit / temperature), keepdims=True, axis=1)
    if use_max:
        ax_probs = np.amax(pred_probs, axis=1)  # using maxium prob to cal entropy
        entropy = (-ax_probs * np.log2(ax_probs + eps))
    else:
        entropy = (-pred_probs * np.log2(pred_probs + eps)).sum(axis=1)
    entropy = np.nan_to_num(entropy)

    if normalize:
        assert use_max is False
        # shannon entropy, entropy has been rescaled to  [0,1]
        entropy = entropy / np.log2(C)
    # entropy[entropy < threhold] = 0.

    if len(entropy.shape) < 3:
        print('check dimensionality of the output')
        raise ValueError
    return entropy
