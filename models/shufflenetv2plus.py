import torch.nn.functional as F
import torch.nn as nn
import torch

from modules import SCSEBlock, InPlaceABN, ASPPInPlaceABNBlock, InPlaceABNWrapper
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init


def conv3x3(in_channels, out_channels, stride=1, bias=True, groups=1, dilate=1):
    """3x3 convolution with padding
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilate, bias=bias, groups=groups, dilation=dilate)


def conv1x1(in_channels, out_channels, groups=1):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3,
                 dilate=1, grouped_conv=True, combine='add', up=False):

        super(ShuffleUnit, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4

        # define the type of ShuffleUnit
        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
            self.dilate = dilate
            self.up = False
            self._combine_func = self._add
        elif self.combine == 'concat':
            # ShuffleUnit Figure 2c
            self.depthwise_stride = 1 if up is True else 2
            self.dilate = dilate if up is True else 1
            self.up = up
            self._combine_func = self._concat

            # ensure output of concat has the same channels as
            # original output channels.
            self.out_channels -= self.in_channels
        else:
            raise ValueError("Cannot combine tensors with \"{}\"" \
                             "Only \"add\" and \"concat\" are" \
                             "supported".format(self.combine))

        # Use a 1x1 grouped or non-grouped convolution to reduce input channels
        # to bottleneck channels, as in a ResNet bottleneck module.
        # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
        self.first_1x1_groups = self.groups if grouped_conv else 1

        self.g_conv_1x1_compress = self._make_grouped_conv1x1(self.in_channels,
                                                              self.bottleneck_channels,
                                                              self.first_1x1_groups,
                                                              batch_norm=True,
                                                              relu=True)

        # 3x3 depthwise convolution followed by batch normalization
        self.depthwise_conv3x3 = conv3x3(self.bottleneck_channels,
                                         self.bottleneck_channels,
                                         stride=self.depthwise_stride,
                                         groups=self.bottleneck_channels,
                                         dilate=self.dilate)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)

        # Use 1x1 grouped convolution to expand from
        # bottleneck_channels to out_channels
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(self.bottleneck_channels,
                                                            self.out_channels,
                                                            self.groups,
                                                            batch_norm=True,
                                                            relu=False)

    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), dim=1)

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups,
                              batch_norm=True, relu=False):

        modules = OrderedDict()

        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self, x):
        # save for combining later with output
        residual = x

        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)
            if self.up is True:
                residual = F.upsample(residual, scale_factor=2, mode="bilinear")

        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)

        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)

        out = self.g_conv_1x1_expand(out)

        out = self._combine_func(residual, out)
        return F.relu(out)


class ShuffleNetV2Plus(nn.Module):
    """ShuffleNet implementation.
    """

    def __init__(self, n_class=19, groups=3, in_channels=3, in_size=(448, 896),
                 out_sec=256, aspp_sec=(12, 24, 36), norm_act=InPlaceABN):
        """ShuffleNet constructor.

        Arguments:
            groups (int, optional): number of groups to be used in grouped
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            num_classes (int, optional): number of classes to predict. Default
                is 19 for ImageNet.

        """
        super(ShuffleNetV2Plus, self).__init__()

        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels = in_channels
        self.n_class = n_class

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))

        # Stage 1 always has 24 output channels
        self.conv1 = conv3x3(self.in_channels,
                             self.stage_out_channels[1],  # stage 1
                             stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2
        self.stage2 = self._make_stage(2, dilate=2)
        # Stage 3
        self.stage3 = self._make_stage(3, dilate=4)
        # Stage 4
        self.stage4 = self._make_stage(4, dilate=8)

        # building last several layers
        self.last_channel = (2 * self.stage_out_channels[1] +
                             self.stage_out_channels[2] +
                             self.stage_out_channels[3] +
                             self.stage_out_channels[4])
        self.out_se = nn.Sequential(SCSEBlock(channel=self.last_channel, reduction=16))

        if self.n_class != 0:
            self.aspp = nn.Sequential(ASPPInPlaceABNBlock(self.last_channel, out_sec,
                                                          feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)),
                                                          aspp_sec=aspp_sec, norm_act=norm_act))

            in_stag2_up_chs = 2 * self.stage_out_channels[1]
            self.score_se = nn.Sequential(SCSEBlock(channel=out_sec + in_stag2_up_chs, reduction=16))
            self.score = nn.Sequential(OrderedDict([("norm.1", norm_act(out_sec + in_stag2_up_chs)),
                                                    ("conv.1", nn.Conv2d(out_sec + in_stag2_up_chs,
                                                                         out_sec + in_stag2_up_chs,
                                                                         kernel_size=3, stride=1, padding=2,
                                                                         dilation=2, bias=False)),
                                                    ("norm.2", norm_act(out_sec + in_stag2_up_chs)),
                                                    ("conv.2", nn.Conv2d(out_sec + in_stag2_up_chs, self.n_class,
                                                                         kernel_size=1, stride=1, padding=0,
                                                                         bias=True)),
                                                    ("up1", nn.Upsample(size=in_size, mode='bilinear'))]))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def _make_stage(self, stage, dilate=1):
        modules = OrderedDict()
        stage_name = "ShuffleUnit_Stage{}".format(stage)

        # First ShuffleUnit in the stage
        # 1. non-grouped 1x1 convolution (i.e. pointwise convolution)
        #   is used in Stage 2. Group convolutions used everywhere else.
        grouped_conv = stage > 2

        # 2. concatenation unit is always used.
        if stage >= 3:
            first_module = ShuffleUnit(self.stage_out_channels[stage - 1],
                                       self.stage_out_channels[stage],
                                       groups=self.groups,
                                       dilate=dilate,
                                       grouped_conv=grouped_conv,
                                       combine='concat',
                                       up=True)
        else:
            first_module = ShuffleUnit(self.stage_out_channels[stage - 1],
                                       self.stage_out_channels[stage],
                                       groups=self.groups,
                                       dilate=1,
                                       grouped_conv=grouped_conv,
                                       combine='concat',
                                       up=False)

        modules[stage_name + "_0"] = first_module

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeats[stage - 2]):
            name = stage_name + "_{}".format(i + 1)
            module = ShuffleUnit(self.stage_out_channels[stage],
                                 self.stage_out_channels[stage],
                                 groups=self.groups,
                                 dilate=dilate,
                                 grouped_conv=True,
                                 combine='add', up=False)
            modules[name] = module

        return nn.Sequential(modules)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # channel_shuffle: shuffle channels in groups
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    @staticmethod
    def _channel_shuffle(x, groups):
        """
            Channel shuffle operation
            :param x: input tensor
            :param groups: split channels into groups
            :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batch_size, groups, channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)

        return x

    def forward(self, x):
        stg0 = self.conv1(x)       # [24, H/2, W/2]
        stg1 = self.maxpool(stg0)  # [24, H/4, W/4]

        stg2 = self.stage2(stg1)   # [240, H/8, W/8]
        stg3 = self.stage3(stg2)   # [480, H/8, W/8]
        stg4 = self.stage4(stg3)   # [960, H/8, W/8]

        stg1_1 = F.avg_pool2d(input=stg0, kernel_size=3, stride=2, padding=1)    # 1/4
        stg1_2 = F.avg_pool2d(input=stg1_1, kernel_size=3, stride=2, padding=1)  # 1/8
        stg1_3 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, padding=1)    # 1/8

        # global average pooling layer
        # (N, 24+24+240+480+960, 56,  112)  1/8
        stg5 = self.out_se(torch.cat([stg2, stg3, stg4, stg1_2, stg1_3], dim=1))

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Decoder: multi-scale feature fusion
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if self.n_class != 0:
            # (N, 24+240+480+960, H/8, W/8) -> (N, 256, H/4, W/4)
            de_stg1 = self.aspp(stg5)[1]

            # (N, 256+24+24, H/4, W/4)
            de_stg1 = self.score_se(torch.cat([de_stg1, stg1, stg1_1], dim=1))

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 3. Classifier: pixel-wise classification-segmentation
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            net_out = self.score(de_stg1)

            return net_out
        else:
            return stg5


if __name__ == "__main__":
    import os
    import time
    from scripts.loss import *
    from functools import partial

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

    net_h, net_w = 384, 768
    model = ShuffleNetV2Plus(n_class=19, groups=3, in_channels=3, in_size=(net_h, net_w),
                             out_sec=256, aspp_sec=(12, 24, 36),
                             norm_act=partial(InPlaceABNWrapper, activation="leaky_relu", slope=0.1)).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    model_dict = model.state_dict()

    pre_weight = torch.load("/zfs/zhang/TrainLog/weights/shufflenet.pth.tar")
    pre_weight = pre_weight["state_dict"]

    pretrained_dict = {"module." + k: v for k, v in pre_weight.items() if "module." + k in model_dict}
    model_dict.update(pretrained_dict)
    state = {'model_state': model_dict}
    torch.save(state, "/zfs/zhang/TrainLog/weights/shufflenetv2plus_model.pkl")
    model.load_state_dict(model_dict)

    del pre_weight
    del model_dict
    del pretrained_dict

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.90, weight_decay=5e-4)
    loss_fn = bootstrapped_cross_entropy2d

    i = 0
    while True:
        i += 1
        print("iter :", i)
        model.train()

        dummy_input = Variable(torch.rand(1, 3, net_h, net_w).cuda(), requires_grad=True)
        dummy_target = Variable(torch.rand(1, net_h, net_w).cuda(), requires_grad=False).long()

        start_time = time.time()
        dummy_out = model(dummy_input)
        print("> Inference Time: {}".format(time.time() - start_time))

        optimizer.zero_grad()

        topk = 512 * 256
        loss = loss_fn(dummy_out, dummy_target, K=topk)
        print("> Loss: {}".format(loss.data[0]))

        loss.backward()
        optimizer.step()

