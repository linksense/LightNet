import torch.nn.functional as F
import torch.nn as nn
import torch


class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)

        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)

        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss


def cross_entropy2d(input, target, weight=None, size_average=True):
    # 1. input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()

    # 2. log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)

    # 3. log_p: (n*h*w, c) - contiguous() required if transpose() is used before view().
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    # 4. target: (n*h*w,)
    mask = target >= 0
    target = target[mask]

    loss = F.nll_loss(log_p, target, ignore_index=250, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
        # loss /= mask.sum().data[0]
    return loss


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=False):
    """A categorical cross entropy loss for 4D tensors.
        We assume the following layout: (batch, classes, height, width)
        Args:
            input: The outputs.
            target: The predictions.
            K: The number of pixels to select in the bootstrapping process.
               The total number of pixels is determined as 512 * multiplier.
        Returns:
            The pixel-bootstrapped cross entropy loss.
    """
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=False):
        n, c, h, w = input.size()

        # 1. The log softmax. log_p: (n, c, h, w)
        log_p = F.log_softmax(input, dim=1)

        # 2. log_p: (n*h*w, c) - contiguous() required if transpose() is used before view().
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        # 3. target: (n*h*w,)
        mask = target >= 0
        target = target[mask]

        loss = F.nll_loss(log_p, target, weight=weight, ignore_index=250,
                          reduce=False, size_average=size_average)

        # For each element in the batch, collect the top K worst predictions
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(input=torch.unsqueeze(input[i], 0),
                                           target=torch.unsqueeze(target[i], 0),
                                           K=K,
                                           weight=weight,
                                           size_average=size_average)
    return loss / float(batch_size)
