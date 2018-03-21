import torch
import torch.nn as nn

from functools import partial
from collections import OrderedDict
from modules import IdentityResidualBlock, SEBlock, ASPPInPlaceABNBlock, ABN


class SEWiderResNetV1(nn.Module):
    def __init__(self, structure, norm_act=ABN, classes=0, dilation=False, is_se=True,
                 in_size=(448, 896), out_sec=(512, 256, 128), aspp_sec=(12, 24, 36)):
        """
        Wider ResNet with pre-activation (identity mapping) and Squeeze & Excitation(SE) blocks

        :param structure: (list of int) Number of residual blocks in each of the six modules of the network.
        :param norm_act:  (callable) Function to create normalization / activation Module.
        :param classes:   (int) Not `0` for segmentation task
        :param dilation:  (bool) `True` for segmentation task
        :param is_se:     (bool) Use Squeeze & Excitation (SE) or not
        :param in_size:   (tuple of int) Size of the input image
        :param out_sec:   (tuple of int) Number of channels of the ASPP output
        :param aspp_sec:  (tuple of int) Dilation rate used in ASPP
        """
        super(SEWiderResNetV1, self).__init__()
        self.structure = structure
        self.dilation = dilation
        self.classes = classes

        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        # Initial layers
        self.mod1 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        ]))

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id == 4:
                        dil = 4
                    elif mod_id == 5:
                        dil = 8
                    else:
                        dil = 1

                    stride = 2 if block_id == 0 and mod_id == 2 else 1

                if mod_id == 4:
                    drop = partial(nn.Dropout2d, p=0.2)
                elif mod_id == 5:
                    drop = partial(nn.Dropout2d, p=0.3)
                else:
                    drop = None

                blocks.append((
                    "block%d" % (block_id + 1),
                    IdentityResidualBlock(in_channels, channels[mod_id], norm_act=norm_act,
                                          stride=stride, dilation=dil, dropout=drop, is_se=is_se)
                ))

                # Update channels and p_keep
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id < 2:
                self.add_module("pool%d" % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)

        if classes != 0:
            self.aspp1 = nn.Sequential(OrderedDict([("aspp1", ASPPInPlaceABNBlock(channels[5][2], out_sec[0],
                                                     feat_res=(int(in_size[0] / 8), int(in_size[1] / 8)),
                                                     aspp_sec=aspp_sec))]))

            self.aspp2 = nn.Sequential(OrderedDict([("aspp2", ASPPInPlaceABNBlock(out_sec[0]+256, out_sec[1],
                                                     feat_res=(int(in_size[0] / 4), int(in_size[1] / 4)),
                                                      aspp_sec=(aspp_sec[0]*2, aspp_sec[1]*2, aspp_sec[2]*2)))]))

            self.aspp3 = nn.Sequential(OrderedDict([("aspp3", ASPPInPlaceABNBlock(out_sec[1]+128, out_sec[2],
                                                    feat_res=(int(in_size[0] / 2), int(in_size[1] / 2)),
                                                    aspp_sec=(aspp_sec[0]*4, aspp_sec[1]*4, aspp_sec[2]*4)))]))

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

    def forward(self, img):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Encoder: feature extraction
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        stg1 = self.mod1(img)               # (N, 64, 448, 896)   1/1
        stg2 = self.mod2(self.pool2(stg1))  # (N, 128, 224, 448)  1/2                 3
        stg3 = self.mod3(self.pool3(stg2))  # (N, 256, 112, 224)  1/4                 3
        stg4 = self.mod4(stg3)              # (N, 512, 56, 112)   1/8 Stride=2        6
        stg4 = self.mod5(stg4)              # (N, 1024, 56, 112)  1/8 dilation=2      3
        stg4 = self.mod6(stg4)              # (N, 2048, 56, 112)  1/8 dilation=4      1
        stg4 = self.mod7(stg4)              # (N, 4096, 56, 112)  1/8 dilation=8      1
        stg4 = self.bn_out(stg4)            # (N, 4096, 56, 112)  1/8

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Decoder: multi-scale feature fusion
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if self.classes != 0:
            # (N, 4096, H/8, W/8) -> (N, 512, H/4, W/4)
            out_stg1, de_stg1 = self.aspp1(stg4)

            # (N, 768, H/4, W/4) -> (N, 256, W/2, H/2)
            out_stg2, de_stg2 = self.aspp2(self._channel_shuffle(torch.cat([de_stg1, stg3], dim=1), 3))

            # (N, 384, H/2, W/2) -> (N, 128, H/1, W/1)
            de_stg3 = self.aspp3(self._channel_shuffle(torch.cat([de_stg2, stg2], dim=1), 3))[1]

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 3. Classifier: pixel-wise classification-segmentation
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            out_stg1 = self.score1(out_stg1)
            out_stg2 = self.score2(out_stg2)
            out_stg3 = self.score3(de_stg3)
            out_stg4 = self.score4(torch.max(torch.max(out_stg1, out_stg2), out_stg3))

            return out_stg1, out_stg2, out_stg3, out_stg4
        else:
            return stg4


_NETS = {
    "16": {"structure": [1, 1, 1, 1, 1, 1]},
    "20": {"structure": [1, 1, 1, 3, 1, 1]},
    "38": {"structure": [3, 3, 6, 3, 1, 1]},
}
