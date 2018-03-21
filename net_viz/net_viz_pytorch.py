from functools import partial

import torch
from torch.autograd import Variable

from models.sewrnetv1 import SEWiderResNetV1
from modules import InPlaceABNWrapper
from net_viz.visualize import make_dot

if __name__ == "__main__":
    net_h, net_w = 448, 896
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 1. Setting up Model...")
    model = SEWiderResNetV1(structure=[3, 3, 6, 3, 1, 1],
                            norm_act=partial(InPlaceABNWrapper, activation="leaky_relu", slope=0.1),
                            classes=19, dilation=True, is_se=True, in_size=(net_h, net_w),
                            out_sec=(512, 256, 128), aspp_sec=(12, 24, 36))
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    pre_weight = torch.load("/media/datavolume3/huijun/SEDPShuffleNet/weights/{}".format(
        "cityscapes_sewrnet_best_model.pkl"))['model_state']
    model.load_state_dict(pre_weight)
    model_dict = model.state_dict()

    for name, param in model.named_parameters():
        print("Name: {}, Size: {}".format(name, param.size()))

    dummy_input = Variable(torch.rand(1, 3, net_h, net_w).cuda(), requires_grad=True)
    output = model(dummy_input)[3]

    graph = make_dot(output, model_dict)
    graph.view()
