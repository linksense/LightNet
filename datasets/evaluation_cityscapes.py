import time
import os

import torch.nn.functional as F
import scipy.misc as misc
import numpy as np
import torch

from datasets.cityscapes_loader import CityscapesLoader
from models.rfmobilenetv2plus import RFMobileNetV2Plus
from models.mobilenetv2plus import MobileNetV2Plus
from models.sewrnetv2 import SEWiderResNetV2
from modules import InPlaceABNWrapper
from torch.autograd import Variable
from functools import partial


def evaluate_valset(data_root, model_path, result_path, traval):
    train_id2label_id = {0: 7,
                         1: 8,
                         2: 11,
                         3: 12,
                         4: 13,
                         5: 17,
                         6: 19,
                         7: 20,
                         8: 21,
                         9: 22,
                         10: 23,
                         11: 24,
                         12: 25,
                         13: 26,
                         14: 27,
                         15: 28,
                         16: 31,
                         17: 32,
                         18: 33}

    net_h, net_w = 896, 1792

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 1. Setup DataLoader
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 0. Setting up DataLoader...")
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    loader = CityscapesLoader(data_root, gt="gtFine", is_transform=True, split='val',
                              img_size=(net_h, net_w), augmentations=None)
    n_classes = loader.n_classes

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 2. Setup Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 1. Setting up Model...")
    model = RFMobileNetV2Plus(n_class=n_classes, in_size=(net_h, net_w), width_mult=1.0,
                            out_sec=256, aspp_sec=(12*2, 24*2, 36*2),
                            norm_act=partial(InPlaceABNWrapper, activation="leaky_relu", slope=0.1))

    # np.arange(torch.cuda.device_count())
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    # state = convert_state_dict(torch.load("/media/datavolume3/huijun/SEDPShuffleNet/weights/{}".format(
    #     args.model_path))['model_state'])
    pre_weight = torch.load(model_path)
    pre_weight = pre_weight['model_state']
    model.load_state_dict(pre_weight)
    del pre_weight

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 2. Inference Model
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    vali_root = "/zfs/zhang/Cityscapes/leftImg8bit/{}".format(traval)
    org_vali_sub = os.listdir(vali_root)
    org_vali_sub.sort()

    for v_id in np.arange(len(org_vali_sub)):
        print("> 2. Processing City # {}...".format(org_vali_sub[v_id]))
        curr_city_path = os.path.join(vali_root, org_vali_sub[v_id])
        images_name = os.listdir(curr_city_path)
        images_name.sort()

        for img_id in np.arange(len(images_name)):
            curr_image = images_name[img_id]
            print("> Processing City #{} Image: {}...".format(org_vali_sub[v_id], curr_image))

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 2.1 Pre-processing Image
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            curr_img_path = os.path.join(curr_city_path, curr_image)
            image = misc.imread(curr_img_path)
            image = np.array(image, dtype=np.uint8)
            image = misc.imresize(image, (loader.img_size[0], loader.img_size[1]))

            image = image[:, :, ::-1]         # From RGB to BGR
            image = image.astype(float)
            image -= loader.mean
            image /= 255.0
            image = image.transpose(2, 0, 1)  # From H*W*C to C*H*W

            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image).float()

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 2.2 Prediction/Inference
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            model.eval()

            start_time = time.time()
            images = Variable(image.cuda(), volatile=True)
            outputs = F.softmax(model(images), dim=1)
            pred = outputs.data.max(1)[1]
            print("> Inference Time: {}s".format(time.time() - start_time))
            pred = np.squeeze(pred.cpu().numpy(), axis=0)

            fun_classes = np.unique(pred)
            print('> {} Classes found before resize: {}'.format(len(fun_classes), fun_classes))
            pred = pred.astype(np.uint8)
            pred = misc.imresize(pred, (1024, 2048), "nearest")

            fun_classes = np.unique(pred)
            print('> {} Classes found after resize: {}'.format(len(fun_classes), fun_classes))

            mapper = lambda t: train_id2label_id[t]
            vfunc = np.vectorize(mapper)
            pred = vfunc(pred)

            fun_classes = np.unique(pred)
            print('> {} Classes found after resize: {}'.format(len(fun_classes), fun_classes))
            print("> Processed City #{} Image: {}, Time: {}s".format(org_vali_sub[v_id], curr_image,
                                                                     (time.time() - start_time)))

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 2.3 Saving prediction result
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            save_path = os.path.join(result_path, org_vali_sub[v_id])
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            # cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
            # cv2.imshow("Prediction", pred)
            # cv2.waitKey(0)
            pred = pred.astype(np.uint8)
            save_name = os.path.basename(curr_image)[:-15] + 'pred_labelIds.png'
            save_path = os.path.join(save_path, save_name)
            misc.imsave(save_path, pred)
            # cv2.imwrite(save_path, pred)

    print("> # +++++++++++++++++++++++++++++++++++++++ #")
    print("> Done!!!")
    print("> # +++++++++++++++++++++++++++++++++++++++ #")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    traval = "test"

    data_path = "/zfs/zhang/Cityscapes"
    result_path = "/zfs/zhang/Cityscapes/results/{}".format(traval)
    model_weight = "/zfs/zhang/TrainLog/weights/cityscapes_rfmobilenetv2_best_model.pkl"
    evaluate_valset(data_path, model_weight, result_path, traval)
