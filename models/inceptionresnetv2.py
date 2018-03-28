"""
    InceptionResNetV2 using the architecture of InceptionV4 as the paper
     "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"
"""

import torch
import torch.nn as nn

from functools import partial
from collections import OrderedDict
from modules import ModifiedSCSEBlock, ASPPInPlaceABNBlock, ABN, InPlaceABNWrapper


class BasicConv2d(nn.Module):
    """
        Define the basic conv-bn-relu block
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilate=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilate, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,      # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):
    """
        Define the 35x35 grid modules of Inception V4 network
        to replace later-half stem modules of InceptionResNet V2
    """
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):
    """
        The 35x35 grid modules of InceptionResNet V2
    """
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):
    """
        The 35x35 to 17x17 reduction module
    """
    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=1, padding=2, dilate=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=2, dilate=2),
            BasicConv2d(256, 384, kernel_size=3, stride=1, padding=2, dilate=2)
        )

        self.branch2 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                     nn.Upsample(scale_factor=2, mode="bilinear"))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):
    """
        The 17x17 grid modules of InceptionResNet V2
    """
    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 6), dilate=(1, 2)),
            BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(6, 0), dilate=(2, 1))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):
    """
        The 17x17 to 8x8 reduction module
    """
    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=1, padding=4, dilate=4)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=4, dilate=4)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=4, dilate=4),
            BasicConv2d(288, 320, kernel_size=3, stride=1, padding=4, dilate=4)
        )

        self.branch3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                     nn.Upsample(scale_factor=2, mode="bilinear"))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):
    """
        The 8x8 grid modules of InceptionResNet V2
    """
    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 4), dilate=(1, 4)),
            BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(4, 0), dilate=(4, 1))
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Module):
    def __init__(self, num_clases=19, in_size=(448, 896), aspp_out=512, fusion_out=64, aspp_sec=(12, 24, 36), norm_act=ABN):
        super(InceptionResNetV2, self).__init__()
        self.num_clases = num_clases
        # Modules
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2, padding=1)

        self.mixed_5b = Mixed_5b()

        self.repeat = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )
        self.mixed_6a = Mixed_6a()

        self.repeat_1 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )
        self.mixed_7a = Mixed_7a()

        self.repeat_2 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )

        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)

        if num_clases != 0:
            self.stg3_fusion = nn.Conv2d(192, fusion_out, kernel_size=1, stride=1, padding=0, bias=False)

            self.aspp = nn.Sequential(OrderedDict([("aspp", ASPPInPlaceABNBlock(1536, aspp_out,
                                                    feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)),
                                                    up_ratio=2, aspp_sec=aspp_sec))]))

            self.score_se = nn.Sequential(ModifiedSCSEBlock(channel=aspp_out+fusion_out, reduction=16))
            self.score = nn.Sequential(OrderedDict([("conv", nn.Conv2d(aspp_out+fusion_out, num_clases,
                                                                       kernel_size=3, stride=1,
                                                                       padding=1, bias=True)),
                                                    ("up", nn.Upsample(size=in_size, mode='bilinear'))]))

    def forward(self, x):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Encoder: feature extraction
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        c1 = self.conv2d_1a(x)    # [1, 32, 224, 448]   1 / 2
        c1 = self.conv2d_2a(c1)   # [1, 32, 224, 448]
        c1 = self.conv2d_2b(c1)   # [1, 64, 224, 448]
        c1 = self.maxpool_3a(c1)  # [1, 64, 112, 224]
        c1 = self.conv2d_3b(c1)   # [1, 80, 112, 224]
        c1 = self.conv2d_4a(c1)   # [1, 192, 112, 224]  1 / 4
        c2 = self.maxpool_5a(c1)  # [1, 192, 56, 112]
        c2 = self.mixed_5b(c2)    # [1, 320, 56, 112]
        c2 = self.repeat(c2)      # [1, 320, 56, 112]    1 / 8
        c2 = self.mixed_6a(c2)    # [1, 1088, 56, 112]
        c2 = self.repeat_1(c2)    # [1, 1088, 56, 112]   1 / 16
        c2 = self.mixed_7a(c2)    # [1, 2080, 56, 112]
        c2 = self.repeat_2(c2)    # [1, 2080, 56, 112]
        c2 = self.block8(c2)      # [1, 2080, 56, 112]
        c2 = self.conv2d_7b(c2)   # [1, 1536, 56, 112]   1 / 32

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Decoder: multi-scale feature fusion
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if self.num_clases != 0:
            # (N, 4096, H/8, W/8) -> (N, 512, H/4, W/4)
            c2 = self.score_se(torch.cat([self.aspp(c2)[1], self.stg3_fusion(c1)], dim=1))

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 3. Classifier: pixel-wise classification-segmentation
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

            return self.score(c2)
        else:
            return c2


'''
TEST
'''
if __name__ == '__main__':
    import time
    from torch.autograd import Variable
    model = InceptionResNetV2().cuda()

    input_model = Variable(torch.randn(1, 3, 448, 896).cuda(), requires_grad=True)

    while True:
        start_time = time.time()
        _ = model(input_model)
        end_time = time.time()
        print("InceptionResNetV2 inference time: {}s".format(end_time - start_time))

    # for name, param in model.named_parameters():
    #     print("Name: {}, Size: {}".format(name, param.size()))
