import torch
import torch.nn as nn

from collections import OrderedDict
from modules import ABN, SEBlock, CatInPlaceABN, ASPPInPlaceABNBlock, DualPathInPlaceABNBlock


class SEDPNShuffleNet(nn.Module):
    def __init__(self, small=False, classes=19, in_size=(448, 896), num_init_features=64,
                 k_r=96, groups=4, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
                 out_sec=(512, 256, 128), dil_sec=(1, 1, 1, 2, 4),
                 aspp_sec=(7, 14, 21), norm_act=ABN):
        super(SEDPNShuffleNet, self).__init__()
        bw_factor = 1 if small else 4

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. conv1 (N, 3, W, H)->(N, 64, W/4, H/4)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if small:
            self.encode_in = nn.Sequential(OrderedDict([("conv_in", nn.Conv2d(3, num_init_features,
                                                       kernel_size=3, stride=2,
                                                       padding=1, bias=False)),
                                                       ("bn_in", norm_act(num_init_features)),
                                                       ("pool_in", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        else:
            self.encode_in = nn.Sequential(OrderedDict([("conv_in", nn.Conv2d(3, num_init_features,
                                                       kernel_size=7, stride=2,
                                                       padding=3, bias=False)),
                                                       ("bn_in", norm_act(num_init_features)),
                                                       ("pool_in", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. conv2 (N, 64, W/4, H/4)->(N, 336, W/4, H/4)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        conv1x1c_ch = 64 * bw_factor                           # For 1x1c ch=64 OR 256 + inc
        inc = inc_sec[0]                                       # For Dense ch=16
        conv1x1a_ch = (k_r * conv1x1c_ch) // (64 * bw_factor)  # For 1x1a ch=96
        conv3x3b_ch = conv1x1a_ch                              # For 3x3b ch=96

        encode_blocks1 = OrderedDict()
        encode_blocks1['conv2_1'] = DualPathInPlaceABNBlock(num_init_features, conv1x1a_ch, conv3x3b_ch,
                                                            conv1x1c_ch, inc, groups, dil_sec[0],
                                                            'proj', norm_act=norm_act)

        in_chs = conv1x1c_ch + 3 * inc                         # 96+3*16=144
        for i in range(2, k_sec[0] + 1):
            encode_blocks1['conv2_' + str(i)] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch,
                                                                        conv1x1c_ch, inc, groups, dil_sec[0],
                                                                        'normal', norm_act=norm_act)
            in_chs += inc

        self.encode_stg1 = nn.Sequential(encode_blocks1)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 3. conv3 (N, 336, W/4, H/4)->(N, 704, W/8, H/8)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        conv1x1c_ch = 128 * bw_factor                          # For 1x1c ch=128 OR 512 + inc
        inc = inc_sec[1]                                       # For Dense ch=32
        conv1x1a_ch = (k_r * conv1x1c_ch) // (64 * bw_factor)  # For 1x1a ch=192
        conv3x3b_ch = conv1x1a_ch                              # For 3x3b ch=192

        encode_blocks2 = OrderedDict()
        encode_blocks2['conv3_1'] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch,
                                                            conv1x1c_ch, inc, groups, dil_sec[1],
                                                            'down', norm_act=norm_act)

        in_chs = conv1x1c_ch + 3 * inc
        for i in range(2, k_sec[1] + 1):
            encode_blocks2['conv3_' + str(i)] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch,
                                                                        conv1x1c_ch, inc, groups, dil_sec[1],
                                                                        'normal', norm_act=norm_act)
            in_chs += inc

        self.encode_stg2 = nn.Sequential(encode_blocks2)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 4. conv4 (N, 704, W/8, H/8)->(N, 1552, W/16, H/16)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        conv1x1c_ch = 256 * bw_factor                          # For 1x1c ch=256 OR 1024 + inc
        inc = inc_sec[2]                                       # For Dense ch=24
        conv1x1a_ch = (k_r * conv1x1c_ch) // (64 * bw_factor)  # For 1x1a ch=384
        conv3x3b_ch = conv1x1a_ch                              # For 3x3b ch=384

        encode_blocks3 = OrderedDict()
        encode_blocks3['conv4_1'] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch,
                                                            conv1x1c_ch, inc, groups, dil_sec[2],
                                                            'down', norm_act=norm_act)

        in_chs = conv1x1c_ch + 3 * inc
        for i in range(2, int(k_sec[2]/2) + 1):
            encode_blocks3['conv4_' + str(i)] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch,
                                                                        conv1x1c_ch, inc, groups, dil_sec[2],
                                                                        'normal', norm_act=norm_act)
            in_chs += inc

        for i in range(int(k_sec[2]/2) + 1, k_sec[2] + 1):
            encode_blocks3['conv4_' + str(i)] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch,
                                                                        conv1x1c_ch, inc, groups, dil_sec[3],
                                                                        'normal', norm_act=norm_act)
            in_chs += inc

        self.encode_stg3 = nn.Sequential(encode_blocks3)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 5. conv5 (N, 1552, W/16, H/16)->(N, 2688, W/16, H/16)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        conv1x1c_ch = 512 * bw_factor                          # For 1x1c ch=512 OR 2048 + inc
        inc = inc_sec[3]                                       # For Dense ch=128
        conv1x1a_ch = (k_r * conv1x1c_ch) // (64 * bw_factor)  # For 1x1a ch=768
        conv3x3b_ch = conv1x1a_ch                              # For 3x3b ch=768

        encode_blocks4 = OrderedDict()
        encode_blocks4['conv5_1'] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch,
                                                            conv1x1c_ch, inc, groups, dil_sec[4],
                                                            'proj', norm_act=norm_act)

        in_chs = conv1x1c_ch + 3 * inc
        for i in range(2, k_sec[3] + 1):
            encode_blocks4['conv5_' + str(i)] = DualPathInPlaceABNBlock(in_chs, conv1x1a_ch, conv3x3b_ch,
                                                                        conv1x1c_ch, inc, groups, dil_sec[4],
                                                                        'normal', norm_act=norm_act)
            in_chs += inc

        encode_blocks4['conv5_bn_ac'] = CatInPlaceABN(in_chs)
        self.encode_stg4 = nn.Sequential(encode_blocks4)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 6. ASPP #1 (N, 2688, W/16, H/16)->(N, 512, W/8, H/8)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        self.aspp1 = nn.Sequential(OrderedDict([("aspp1", ASPPInPlaceABNBlock(in_chs, out_sec[0],
                                                 feat_res=(int(in_size[0] / 16), int(in_size[1] / 16)),
                                                 aspp_sec=aspp_sec, norm_act=norm_act))]))

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 7. ASPP #2 (N, 1216, W/8, H/8)->(N, 256, W/4, H/4)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        self.aspp2_in = nn.Sequential(OrderedDict([("aspp2_in", CatInPlaceABN(704, norm_act=norm_act))]))
        self.aspp2 = nn.Sequential(OrderedDict([("aspp2", ASPPInPlaceABNBlock(out_sec[0]+704, out_sec[1],
                                                 feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)),
                                                 aspp_sec=(aspp_sec[0]*2, aspp_sec[1]*2, aspp_sec[2]*2),
                                                 norm_act=norm_act))]))

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 8. ASPP #3 (N, 592, W/4, H/4)->(N, 128, W/1, H/1)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        self.aspp3_in = nn.Sequential(OrderedDict([("aspp3_in", CatInPlaceABN(336, norm_act=norm_act))]))
        self.aspp3 = nn.Sequential(OrderedDict([("aspp3", ASPPInPlaceABNBlock(out_sec[1]+336, out_sec[2],
                                                feat_res=(int(in_size[0] / 4), int(in_size[1] / 4)), up_ratio=4,
                                                aspp_sec=(aspp_sec[0]*4, aspp_sec[1]*4, aspp_sec[2]*4),
                                                norm_act=norm_act))]))

        self.score1 = nn.Sequential(OrderedDict([("score1", nn.Conv2d(out_sec[0], classes,
                                                                      kernel_size=1, stride=1, padding=0, bias=True)),
                                                 ("se1_classes", SEBlock(classes, 4)),
                                                 ("up1", nn.Upsample(size=in_size, mode='bilinear'))]))

        self.score2 = nn.Sequential(OrderedDict([("score2", nn.Conv2d(out_sec[1], classes,
                                                                      kernel_size=1, stride=1, padding=0, bias=True)),
                                                ("se2_classes", SEBlock(classes, 4)),
                                                ("up2", nn.Upsample(size=in_size, mode='bilinear'))]))

        self.score3 = nn.Sequential(OrderedDict([("score3", nn.Conv2d(out_sec[2], classes,
                                                                      kernel_size=1, stride=1, padding=0, bias=True)),
                                                ("se3_classes", SEBlock(classes, 4))]))

        self.score4 = nn.Sequential(OrderedDict([("score4_norm", norm_act(classes)),
                                                ("score4", nn.Conv2d(classes, classes,
                                                                     kernel_size=1, stride=1, padding=0, bias=True)),
                                                ("se4_classes", SEBlock(classes, 4))]))

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
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Encoder: feature extraction
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        en_in = self.encode_in(x)            # (N, 64, W/4, H/4)
        en_stg1 = self.encode_stg1(en_in)    # (N, 336, W/4, H/4)
        en_stg2 = self.encode_stg2(en_stg1)  # (N, 704, W/8, H/8)
        en_stg3 = self.encode_stg3(en_stg2)  # (N, 1552, W/16, H/16)
        en_stg4 = self.encode_stg4(en_stg3)  # (N, 2688, W/16, H/16)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Decoder: multi-scale feature fusion
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        out_stg1, de_stg1 = self.aspp1(en_stg4)                                       # (N, 512, W/8, H/8)

        # (N, 256, W/4, H/4)
        out_stg2, de_stg2 = self.aspp2(self._channel_shuffle(torch.cat([de_stg1, self.aspp2_in(en_stg2)], dim=1), 2))

        # (N, 128, W/1, H/1)
        de_stg3 = self.aspp3(self._channel_shuffle(torch.cat([de_stg2, self.aspp3_in(en_stg1)], dim=1), 2))[1]

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 3. Classifier: pixel-wise classification-segmentation
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        out_stg1 = self.score1(out_stg1)
        out_stg2 = self.score2(out_stg2)
        out_stg3 = self.score3(de_stg3)
        out_stg4 = self.score4(torch.max(torch.max(out_stg1, out_stg2), out_stg3))
        return out_stg1, out_stg2, out_stg3, out_stg4
