import torch
import os

from models.sedpshufflenet import SEDPNShuffleNet
from scripts.loss import cross_entropy2d
from modules import InPlaceABNWrapper
from torch.autograd import Variable
from functools import partial

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    model = SEDPNShuffleNet(small=False, classes=19, in_size=(448, 896), num_init_features=64,
                            k_r=96, groups=4, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
                            out_sec=(512, 256, 128), dil_sec=(1, 1, 1, 2, 4), aspp_sec=(6, 12, 18),
                            norm_act=partial(InPlaceABNWrapper, activation="leaky_relu", slope=0.1))

    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    for name, param in model.named_parameters():
        print("Name: {}, Size: {}".format(name, param.size()))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.90, weight_decay=5e-4)
    loss_fn = cross_entropy2d

    i = 0
    while True:
        i += 1
        print("iter :", i)
        model.train()

        dummy_input = Variable(torch.rand(2, 3, 448, 896).cuda(), requires_grad=True)
        dummy_target = Variable(torch.rand(2, 448, 896).cuda(), requires_grad=False).long()
        output = model(dummy_input)[3]

        optimizer.zero_grad()

        loss = loss_fn(output, dummy_target)
        loss.backward()

        optimizer.step()
