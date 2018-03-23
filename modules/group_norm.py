import torch
import torch.nn as nn
import tensorflow as tf
from torch.autograd import Variable


def tf_group_norm(x, gamma, beta, groups, eps=1e-5):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    batch_size, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    x = tf.reshape(x, [batch_size, groups, channels_per_group, height, width])
    mean, var = tf.nn.moments(x, axes=[2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [batch_size, num_channels, height, width])
    return x * gamma + beta


class GroupNorm2D(nn.Module):
    """
    Group Normalization
    Reference: https://128.84.21.199/abs/1803.08494v1
    """
    def __init__(self, gamma, beta, groups, eps=1e-5):
        super(GroupNorm2D, self).__init__()

        self.gamma = gamma
        self.beta = beta
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        # x: input features with shape [N, C, H, W]
        # gamma, beta: scale and offset, with shape [1, C, 1, 1]
        # groups: number of groups for GroupNorm

        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)

        mean = torch.mean(torch.mean(torch.mean(x, dim=2, keepdim=True), dim=3, keepdim=True), dim=4, keepdim=True)
        var = torch.var(torch.var(torch.var(x, dim=2, keepdim=True), dim=3, keepdim=True), dim=4, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)

        # reshape
        x = x.view(batch_size, num_channels, height, width)

        return x * self.gamma + self.beta
