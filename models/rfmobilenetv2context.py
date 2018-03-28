import math
import torch
import encoding
import torch.nn as nn
import torch.nn.functional as F

from modules import RFBlock, ContextEncodeDropInplaceABN
from modules import InPlaceABN,  InPlaceABNWrapper
from modules.misc import InvertedResidual, conv_bn
from collections import OrderedDict
from functools import partial


class MobileNetV2Context(nn.Module):
    def __init__(self, n_class=19, in_size=(448, 896), width_mult=1.,
                 out_sec=256, context=(32, 4), aspp_sec=(12, 24, 36), norm_act=InPlaceABN):
        """
        MobileNetV2Plus: MobileNetV2 based Semantic Segmentation
        :param n_class:    (int)  Number of classes
        :param in_size:    (tuple or int) Size of the input image feed to the network
        :param width_mult: (float) Network width multiplier
        :param out_sec:    (tuple) Number of the output channels of the ASPP Block
        :param context:   (tuple) K and reduction
        """
        super(MobileNetV2Context, self).__init__()

        self.n_class = n_class
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, d
            [1, 16, 1, 1, 1],    # 1/2
            [6, 24, 2, 2, 1],    # 1/4
            [6, 32, 3, 2, 1],    # 1/8
            [6, 64, 4, 1, 2],    # 1/8
            [6, 96, 3, 1, 4],    # 1/8
            [6, 160, 3, 1, 8],   # 1/8
            [6, 320, 1, 1, 16],  # 1/8
        ]

        # building first layer
        assert in_size[0] % 8 == 0
        assert in_size[1] % 8 == 0

        self.input_size = in_size

        input_channel = int(32 * width_mult)
        self.mod1 = nn.Sequential(OrderedDict([("conv1", conv_bn(inp=3, oup=input_channel, stride=2))]))

        # building inverted residual blocks
        mod_id = 0
        for t, c, n, s, d in self.interverted_residual_setting:
            output_channel = int(c * width_mult)

            # Create blocks for module
            blocks = []
            for block_id in range(n):
                if block_id == 0 and s == 2:
                    blocks.append(("block%d" % (block_id + 1), InvertedResidual(inp=input_channel,
                                                                                oup=output_channel,
                                                                                stride=s,
                                                                                dilate=1,
                                                                                expand_ratio=t)))
                else:
                    blocks.append(("block%d" % (block_id + 1), InvertedResidual(inp=input_channel,
                                                                                oup=output_channel,
                                                                                stride=1,
                                                                                dilate=d,
                                                                                expand_ratio=t)))

                input_channel = output_channel

            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1

        # building last several layers
        org_last_chns = (self.interverted_residual_setting[0][1] +
                         self.interverted_residual_setting[1][1] +
                         self.interverted_residual_setting[2][1] +
                         self.interverted_residual_setting[3][1] +
                         self.interverted_residual_setting[4][1] +
                         self.interverted_residual_setting[5][1] +
                         self.interverted_residual_setting[6][1])

        self.last_channel = int(org_last_chns * width_mult) if width_mult > 1.0 else org_last_chns
        self.context1 = ContextEncodeDropInplaceABN(channel=self.last_channel, K=context[0],
                                                    reduction=context[1], norm_act=norm_act)
        self.se_loss1 = nn.Sequential(OrderedDict([("linear", nn.Linear(int(self.last_channel / context[1]) *
                                                                        context[0], self.n_class))]))

        self.rfblock = nn.Sequential(RFBlock(in_chs=self.last_channel, out_chs=out_sec,
                                             scale=1.0, feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)),
                                             up_ratio=2, aspp_sec=aspp_sec, norm_act=norm_act))
        if self.n_class != 0:
            in_stag2_up_chs = self.interverted_residual_setting[1][1] + self.interverted_residual_setting[0][1]

            self.context2 = ContextEncodeDropInplaceABN(channel=(out_sec + in_stag2_up_chs), K=context[0],
                                                        reduction=context[1], norm_act=norm_act)
            self.se_loss2 = nn.Sequential(OrderedDict([("linear", nn.Linear(int((out_sec + in_stag2_up_chs)
                                                                                / context[1]) * context[0],
                                                                            self.n_class))]))

            self.score = nn.Sequential(OrderedDict([("norm.1", norm_act(out_sec + in_stag2_up_chs)),
                                                    ("conv.1", nn.Conv2d(out_sec + in_stag2_up_chs, out_sec,
                                                                         kernel_size=3, stride=1, padding=2,
                                                                         dilation=2, bias=False)),
                                                    ("norm.2", norm_act(out_sec)),
                                                    ("conv.2", nn.Conv2d(out_sec, self.n_class,
                                                                         kernel_size=1, stride=1, padding=0,
                                                                         bias=True)),
                                                    ("up1", nn.Upsample(size=in_size, mode='bilinear'))]))
            """

            self.score = nn.Sequential(OrderedDict([("norm", norm_act(out_sec + in_stag2_up_chs)),
                                                    ("conv", nn.Conv2d(out_sec + in_stag2_up_chs, self.n_class,
                                                                       kernel_size=1, stride=1, padding=0,
                                                                       bias=True)),
                                                    ("up1", nn.Upsample(size=in_size, mode='bilinear'))]))
            """
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Encoder: feature extraction
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        stg1 = self.mod1(x)     # (N, 32,   224, 448)  1/2
        stg1 = self.mod2(stg1)  # (N, 16,   224, 448)  1/2 -> 1/4 -> 1/8
        stg2 = self.mod3(stg1)  # (N, 24,   112, 224)  1/4 -> 1/8
        stg3 = self.mod4(stg2)  # (N, 32,   56,  112)  1/8
        stg4 = self.mod5(stg3)  # (N, 64,   56,  112)  1/8 dilation=2
        stg5 = self.mod6(stg4)  # (N, 96,   56,  112)  1/8 dilation=4
        stg6 = self.mod7(stg5)  # (N, 160,  56,  112)  1/8 dilation=8
        stg7 = self.mod8(stg6)  # (N, 320,  56,  112)  1/8 dilation=16

        stg1_1 = F.max_pool2d(input=stg1, kernel_size=3, stride=2, ceil_mode=True)    # 1/4
        stg1_2 = F.max_pool2d(input=stg1_1, kernel_size=3, stride=2, ceil_mode=True)  # 1/8
        stg2_1 = F.max_pool2d(input=stg2, kernel_size=3, stride=2, ceil_mode=True)    # 1/8

        # (N, 712, 56,  112)  1/8  (16+24+32+64+96+160+320)
        enc1, stg8 = self.context1(torch.cat([stg3, stg4, stg5, stg6, stg7, stg1_2, stg2_1], dim=1))
        stg8 = self.rfblock(stg8)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Decoder: multi-scale feature fusion
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if self.n_class != 0:
            enc2, stg8 = self.context2(torch.cat([stg8, stg2, stg1_1], dim=1))
            return self.se_loss1(enc1), self.se_loss2(enc2), self.score(stg8)
        else:
            return stg8


if __name__ == '__main__':
    import os
    import torch
    from torch.autograd import Variable

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

    dummy_in = Variable(torch.randn(1, 3, 448, 448).cuda(), requires_grad=True)

    model = MobileNetV2Context(n_class=19, in_size=(448, 448), width_mult=1., out_sec=256, context=(32, 4),
                               norm_act=partial(InPlaceABNWrapper, activation="leaky_relu", slope=0.1)).cuda()
    enc1, enc2, dummy_out = model(dummy_in)
    print("ok!!!")

