# from modules import InPlaceABN, InPlaceABNSync
from itertools import filterfalse
import torch.nn as nn
import numpy as np
import torch
import math
import os


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('> Computing mean and std of images in the dataset..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

"""
def init_weights(model, activation="leaky_relu", slope=0.1, init="kaiming_uniform", gain_multiplier=1):
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            init_fn = getattr(nn.init, init)

            if init.startswith("xavier") or init == "orthogonal":
                gain = gain_multiplier

                if activation == "relu" or activation == "elu":
                    gain *= nn.init.calculate_gain("relu")
                elif activation == "leaky_relu":
                    gain *= nn.init.calculate_gain("leaky_relu", slope)

                init_fn(m.weight, gain)
            elif init.startswith("kaiming"):
                if activation == "relu" or activation == "elu":
                    init_fn(m.weight, 0)
                else:
                    init_fn(m.weight, slope)

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant(m.bias, 0.0)

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN) or isinstance(m, InPlaceABNSync):
            nn.init.constant(m.weight, 1.0)
            nn.init.constant(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight, 0.1)
            nn.init.constant(m.bias, 0.0)
"""

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=89280, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    curr_lr = init_lr
    if iter % lr_decay_iter or iter > max_iter:
        return curr_lr

    for param_group in optimizer.param_groups:
        curr_lr = init_lr * (1 - iter / max_iter) ** power
        param_group['lr'] = curr_lr

    return curr_lr


def cosine_annealing_lr(period, batch_idx):
    # returns normalised anytime sgdr schedule given period and batch_idx
    # best performing settings reported in paper are T_0 = 10, T_mult=2
    # so always use T_mult=2
    # \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
    # \cos(\frac{T_{cur}}{T_{max}}\pi))

    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx/restart_period > 1.:
        batch_idx = batch_idx - restart_period
        restart_period = restart_period * 2.

    radians = math.pi*(batch_idx/restart_period)
    return 0.5*(1.0 + math.cos(radians))


def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.0:
            param_group['lr'] = lr
    return optimizer


def poly_topk_scheduler(init_topk, iter, topk_decay_iter=1, max_iter=89280, power=0.9):
    curr_topk = init_topk
    if iter % topk_decay_iter or iter > max_iter:
        return curr_topk

    curr_topk = int(init_topk * (1 - iter / max_iter) ** power)
    if curr_topk <= 128:
        curr_topk = 128

    return curr_topk


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value

    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state

    """

    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        state_dict[name] = v
        del state_dict[k]
    return state_dict


def update_aggregated_weight_average(model, weight_aws, full_iter, cycle_length):
    for name, param in model.named_parameters():
        n_model = full_iter/cycle_length
        weight_aws[name] = (weight_aws[name]*n_model + param.data)/(n_model + 1)

    return weight_aws


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
