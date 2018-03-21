from collections import OrderedDict

import torch
import torch.nn as nn

from .bn import ABN


class DenseModule(nn.Module):
    def __init__(self, in_chns, squeeze_ratio, out_chns, n_layers, dilate_sec=(1, 2, 4, 8, 16), norm_act=ABN):
        super(DenseModule, self).__init__()
        self.n_layers = n_layers
        self.mid_out = int(in_chns * squeeze_ratio)

        self.convs1 = nn.ModuleList()
        self.convs3 = nn.ModuleList()

        for idx in range(self.n_layers):
            dilate = dilate_sec[idx % len(dilate_sec)]
            self.last_channel = in_chns + idx * out_chns

            """
            self.convs1.append(nn.Sequential(OrderedDict([
                ("bn", norm_act(self.last_channel)),
                ("conv", nn.Conv2d(self.last_channel, self.mid_out, 1, bias=False))
            ])))
            """

            self.convs3.append(nn.Sequential(OrderedDict([
                ("bn", norm_act(self.last_channel)),
                ("conv", nn.Conv2d(self.last_channel, out_chns, kernel_size=3, stride=1,
                                   padding=dilate, dilation=dilate, bias=False))
            ])))

    @property
    def out_channels(self):
        return self.last_channel + 1

    def forward(self, x):
        inputs = [x]
        for i in range(self.n_layers):
            x = torch.cat(inputs, dim=1)
            # x = self.convs1[i](x)
            x = self.convs3[i](x)
            inputs += [x]

        return torch.cat(inputs, dim=1)
