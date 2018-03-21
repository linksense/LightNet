import torch.nn.functional as F
import torch.nn as nn
import torch

from modules import SCSEBlock, InPlaceABN, ASPPInPlaceABNBlock, InPlaceABNWrapper, DenseModule
from collections import OrderedDict
from functools import partial


class MixedScaleDenseNet(nn.Module):
    """
    Mixed Scale Dense Network
    """
    def __init__(self, n_class=19, in_size=(448, 896), num_layers=128, in_chns=32, squeeze_ratio=1.0/32, out_chns=1,
                 dilate_sec=(1, 2, 4, 8, 4, 2), aspp_sec=(24, 48, 72), norm_act=InPlaceABN):
        """
        MixedScaleDenseNet: Mixed Scale Dense Network

        :param n_class:    (int) Number of classes
        :param in_size:    (tuple or int) Size of the input image feed to the network
        :param num_layers: (int) Number of layers used in the mixed scale dense block/stage
        :param in_chns:    (int) Input channels of the mixed scale dense block/stage
        :param out_chns:   (int) Output channels of each Conv used in the mixed scale dense block/stage
        :param dilate_sec: (tuple) Dilation rates used in the mixed scale dense block/stage
        :param aspp_sec:   (tuple) Dilation rates used in ASPP
        :param norm_act:   (object) Batch Norm Activation Type
        """
        super(MixedScaleDenseNet, self).__init__()

        self.n_classes = n_class

        self.conv_in = nn.Sequential(OrderedDict([("conv", nn.Conv2d(in_channels=3, out_channels=in_chns,
                                                                     kernel_size=7, stride=2,
                                                                     padding=3, bias=False)),
                                                  ("norm", norm_act(in_chns)),
                                                  ("pool", nn.MaxPool2d(3, stride=2, padding=1))]))

        self.dense = DenseModule(in_chns, squeeze_ratio, out_chns, num_layers,
                                 dilate_sec=dilate_sec, norm_act=norm_act)

        self.last_channel = self.dense.out_channels  # in_chns + num_layers * out_chns

        # Pooling and predictor
        self.feat_out = norm_act(self.last_channel)
        self.out_se = nn.Sequential(SCSEBlock(channel=self.last_channel, reduction=16))

        if self.n_classes != 0:
            self.aspp = nn.Sequential(ASPPInPlaceABNBlock(self.last_channel, self.last_channel,
                                                          feat_res=(int(in_size[0] / 4), int(in_size[1] / 4)),
                                                          aspp_sec=aspp_sec, norm_act=norm_act))

            self.score_se = nn.Sequential(SCSEBlock(channel=self.last_channel, reduction=16))
            self.score = nn.Sequential(OrderedDict([("norm.1", norm_act(self.last_channel)),
                                                    ("conv.1", nn.Conv2d(self.last_channel, self.last_channel,
                                                                         kernel_size=3, stride=1, padding=2,
                                                                         dilation=2, bias=False)),
                                                    ("norm.2", norm_act(self.last_channel)),
                                                    ("conv.2", nn.Conv2d(self.last_channel, self.n_classes,
                                                                         kernel_size=1, stride=1, padding=0,
                                                                         bias=True)),
                                                    ("up1", nn.Upsample(size=in_size, mode='bilinear'))]))

    def forward(self, x):
        # [N, 3, H, W] -> [N, 32, H/4, W/4] -> [N, 128+32, H/4, W/4]  1/4
        x = self.out_se(self.feat_out(self.dense(self.conv_in(x))))

        if self.n_classes != 0:
            return self.score(self.score_se(self.aspp(x)[1]))
        else:
            return x


# +++++++++++++++++++++++++++++++++++++++++++++ #
# Test the code of 'MixedScaleDenseNet'
# +++++++++++++++++++++++++++++++++++++++++++++ #
if __name__ == "__main__":
    import time
    from scripts.loss import *
    from torch.autograd import Variable

    net_h, net_w = 448, 896
    model = MixedScaleDenseNet(n_class=19, in_size=(net_h, net_w), num_layers=96, in_chns=32,
                               squeeze_ratio=1.0 / 32, out_chns=1,
                               dilate_sec=(1, 2, 4, 8, 16, 8, 4, 2),
                               aspp_sec=(24, 48, 72),
                               norm_act=partial(InPlaceABNWrapper, activation="leaky_relu", slope=0.1))
    model.cuda()

    model_dict = model.state_dict()

    pre_weight = torch.load("/zfs/zhang/TrainLog/weights/cityscapes_msdensenet_best_model.pkl")["model_state"]

    keys = list(pre_weight.keys())
    keys.sort()
    for k in keys:
        if "aspp" in k:
            pre_weight.pop(k)
        if "score" in k:
            pre_weight.pop(k)

    state = {"model_state": pre_weight}
    torch.save(state, "{}msdensenet_model.pkl".format("/zfs/zhang/TrainLog/weights/"))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.90, weight_decay=5e-4)
    loss_fn = bootstrapped_cross_entropy2d

    i = 0
    while True:
        i += 1
        print("iter :", i)
        model.train()

        dummy_input = Variable(torch.rand(2, 3, net_h, net_w).cuda(), requires_grad=True)
        dummy_target = Variable(torch.rand(2, net_h, net_w).cuda(), requires_grad=False).long()

        start_time = time.time()
        dummy_out = model(dummy_input)
        print("> Inference Time: {}".format(time.time()-start_time))

        optimizer.zero_grad()

        topk = 512 * 256
        loss = loss_fn(dummy_out, dummy_target, K=topk)
        print("> Loss: {}".format(loss.data[0]))

        loss.backward()
        optimizer.step()

    # model_dict = model.state_dict()
    # graph = make_dot(dummy_out, model_dict)
    # graph.view()
