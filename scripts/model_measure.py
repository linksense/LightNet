import torch
import operator
import torch.nn as nn
from functools import reduce
from torch.autograd import Variable


count_ops = 0
count_params = 0


def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    ### ops_nonlinearity
    elif type_name in ['ReLU', 'ReLU6', 'LeakyReLU', 'Sigmoid']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    ### ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel',
                       'Dropout', 'InPlaceABN', 'InPlaceABNSync', 'Upsample', 'MaxPool2d']:
        delta_params = get_layer_param(layer)

    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def measure_model(model, H, W):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, 3, H, W).cuda())

    def should_measure(x):
        return is_leaf(x) or is_pruned(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops, count_params


if __name__ == '__main__':
    from functools import partial
    from modules import InPlaceABNWrapper
    from models.sewrnetv2 import SEWiderResNetV2
    from models.mobilenetv2plus import MobileNetV2Plus
    from models.shufflenetv2plus import ShuffleNetV2Plus
    from models.rfmobilenetv2plus import RFMobileNetV2Plus
    from models.mixscaledensenet import MixedScaleDenseNet

    net_h, net_w = 448, 896

    """
    model = RFMobileNetV2Plus(n_class=19, in_size=(net_h, net_w), width_mult=1.0,
                            out_sec=256, aspp_sec=(12, 24, 36),
                            norm_act=partial(InPlaceABNWrapper, activation="leaky_relu", slope=0.1))


    model = SEWiderResNetV2(structure=[3, 3, 6, 3, 1, 1],
                            norm_act=partial(InPlaceABNWrapper, activation="leaky_relu", slope=0.1),
                            classes=19, dilation=True, is_se=True, in_size=(net_h, net_w),
                            aspp_out=512, fusion_out=64, aspp_sec=(12, 24, 36))

    model = ShuffleNetV2Plus(n_class=19, groups=3, in_channels=3, in_size=(net_h, net_w),
                             out_sec=256, aspp_sec=(12, 24, 36),
                             norm_act=partial(InPlaceABNWrapper, activation="leaky_relu", slope=0.1))
    """
    model = MixedScaleDenseNet(n_class=19, in_size=(net_h, net_w), num_layers=96, in_chns=32,
                               squeeze_ratio=1.0 / 32, out_chns=1,
                               dilate_sec=(1, 2, 4, 8, 16),
                               aspp_sec=(24, 48, 72),
                               norm_act=partial(InPlaceABNWrapper, activation="leaky_relu", slope=0.1))

    model.cuda()

    count_ops, count_params = measure_model(model, net_h, net_w)
    print('FLOPs: {}, Params: {}'.format(count_ops, count_params))
