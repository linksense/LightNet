from torch.autograd import Variable
import torch.nn.functional as F
import scripts.utils as utils
import torch.nn as nn
import numpy as np
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


class FocalLoss2D(nn.Module):
    """
    Focal Loss, which is proposed in:
        "Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002v2)"
    """
    def __init__(self, num_classes=19, ignore_label=250, alpha=0.25, gamma=2, size_average=True):
        """
        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        :param num_classes:   (int) num of the classes
        :param ignore_label:  (int) ignore label
        :param alpha:         (1D Tensor or Variable) the scalar factor
        :param gamma:         (float) gamma > 0;
                                      reduces the relative loss for well-classified examples (probabilities > .5),
                                      putting more focus on hard, mis-classified examples
        :param size_average:  (bool): By default, the losses are averaged over observations for each mini-batch.
                                      If the size_average is set to False, the losses are
                                      instead summed for each mini-batch.
        """
        super(FocalLoss2D, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.size_average = size_average
        self.one_hot = Variable(torch.eye(self.num_classes))

    def forward(self, cls_preds, cls_targets):
        """

        :param cls_preds:    (n, c, h, w)
        :param cls_targets:  (n, h, w)
        :return:
        """
        assert not cls_targets.requires_grad
        assert cls_targets.dim() == 3
        assert cls_preds.size(0) == cls_targets.size(0), "{0} vs {1} ".format(cls_preds.size(0), cls_targets.size(0))
        assert cls_preds.size(2) == cls_targets.size(1), "{0} vs {1} ".format(cls_preds.size(2), cls_targets.size(1))
        assert cls_preds.size(3) == cls_targets.size(2), "{0} vs {1} ".format(cls_preds.size(3), cls_targets.size(3))

        if cls_preds.is_cuda:
            self.one_hot = self.one_hot.cuda()

        n, c, h, w = cls_preds.size()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. target reshape and one-hot encode
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1.1. target: (n*h*w,)
        cls_targets = cls_targets.view(n * h * w, 1)
        target_mask = (cls_targets >= 0) * (cls_targets != self.ignore_label)

        cls_targets = cls_targets[target_mask]
        cls_targets = self.one_hot.index_select(dim=0, index=cls_targets)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. compute focal loss for multi-classification
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2.1. The softmax. prob: (n, c, h, w)
        prob = F.softmax(cls_preds, dim=1)
        # 2.2. prob: (n*h*w, c) - contiguous() required if transpose() is used before view().
        prob = prob.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        prob = prob[target_mask.repeat(1, c)]
        prob = prob.view(-1, c)  # (n*h*w, c)

        probs = torch.clamp((prob * cls_targets).sum(1).view(-1, 1), min=1e-8, max=1.0)
        batch_loss = -self.alpha * (torch.pow((1 - probs), self.gamma)) * probs.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class SemanticEncodingLoss(nn.Module):
    def __init__(self, num_classes=19, ignore_label=250, alpha=0.25):
        super(SemanticEncodingLoss, self).__init__()
        self.alpha = alpha

        self.num_classes = num_classes
        self.ignore_label = ignore_label

    def unique_encode(self, cls_targets):
        batch_size, _, _ = cls_targets.size()
        target_mask = (cls_targets >= 0) * (cls_targets != self.ignore_label)
        cls_targets = [cls_targets[idx].masked_select(target_mask[idx]) for idx in np.arange(batch_size)]

        # unique_cls = [np.unique(label.numpy(), return_counts=True) for label in cls_targets]
        unique_cls = [np.unique(label.numpy()) for label in cls_targets]

        encode = np.zeros((batch_size, self.num_classes), dtype=np.uint8)

        for idx in np.arange(batch_size):
            np.put(encode[idx], unique_cls[idx], 1)

        return torch.from_numpy(encode).float()

    def forward(self, predicts, enc_cls_target, size_average=True):
        se_loss = F.binary_cross_entropy_with_logits(predicts, enc_cls_target, weight=None,
                                                     size_average=size_average)

        return self.alpha * se_loss


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Lovasz-Softmax
# Maxim Berman 2018 ESAT-PSI KU Leuven
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union

    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = utils.mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(utils.mean, zip(*ious))  # mean accross images if per_image
    return 100 * np.array(ious)


def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = utils.mean(lovasz_softmax_flat(*flatten_probas(prob, lab, ignore), only_present=only_present)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return utils.mean(losses)


def flatten_probas(scores, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = scores.size()
    scores = scores.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vscores, vlabels


if __name__ == "__main__":
    from torch.autograd import Variable

    while True:
        dummy_in = Variable(torch.randn(2, 3, 32, 32), requires_grad=True)
        dummy_gt = Variable(torch.LongTensor(2, 32, 32).random_(0, 3))

        dummy_in = F.softmax(dummy_in, dim=1)
        loss = lovasz_softmax(dummy_in, dummy_gt, ignore=255)
        print(loss.data[0])
