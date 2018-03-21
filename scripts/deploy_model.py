import argparse
import os

import torch.nn.functional as F
import scipy.misc as misc
import numpy as np
import torch
import time
import cv2

from datasets.cityscapes_loader import CityscapesLoader
from models.rfmobilenetv2plus import RFMobileNetV2Plus
from models.mobilenetv2plus import MobileNetV2Plus
from models.sewrnetv2 import SEWiderResNetV2
from modules import InPlaceABNWrapper
from torch.autograd import Variable
from functools import partial

try:
    import pydensecrf.densecrf as dcrf
except:
    print("Failed to import pydensecrf, CRF post-processing will not work")


def test(args):
    net_h, net_w = 896, 1792   # 512, 1024  768, 1536  896, 1792  1024, 2048
    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    img_path = "/afs/cg.cs.tu-bs.de/home/zhang/SEDPShuffleNet/deploy/munster_000168_000019_leftImg8bit.png"
    mask_path = "/afs/cg.cs.tu-bs.de/home/zhang/SEDPShuffleNet/deploy/munster_000168_000019_gtFine_color.png"
    img = misc.imread(img_path)
    msk = cv2.imread(mask_path)

    data_path = "/zfs/zhang/Cityscapes"
    loader = CityscapesLoader(data_path, is_transform=True, split='val',
                              img_size=(net_h, net_w),
                              augmentations=None)
    n_classes = loader.n_classes

    # Setup Model
    print("> 1. Setting up Model...")
    model = MobileNetV2Plus(n_class=n_classes, in_size=(net_h, net_w), width_mult=1.0,
                            out_sec=256, aspp_sec=(12*2, 24*2, 36*2),
                            norm_act=partial(InPlaceABNWrapper, activation="leaky_relu", slope=0.1))

    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    # state = convert_state_dict(torch.load("/media/datavolume3/huijun/SEDPShuffleNet/weights/{}".format(
    #     args.model_path))['model_state'])
    pre_weight = torch.load("/zfs/zhang/TrainLog/weights/{}".format(
        "cityscapes_mobilenetv2_best_model.pkl"))
    pre_weight = pre_weight['model_state']
    model.load_state_dict(pre_weight)

    # state = model.state_dict()
    # torch.save(state, "cityscapes_sewrnet_best_model.pkl")

    resized_img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]), interp='bilinear')

    img = img[:, :, ::-1]
    img = np.array(img, dtype=np.uint8)
    img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
    img = img.astype(np.float32)
    img -= loader.mean
    img = img.astype(np.float32) / 255.0

    # NHWC -> NCWH
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    model.eval()

    images = Variable(img.cuda(), volatile=True)

    start_time = time.time()
    outputs = F.softmax(model(images), dim=1)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    print("Inference time: {}s".format(time.time()-start_time))

    decoded = loader.decode_segmap(pred)*255
    decoded = decoded.astype(np.uint8)
    img_msk = cv2.addWeighted(resized_img, 0.60, decoded, 0.40, 0)
    fun_classes = np.unique(pred)
    print('> {} Classes found: {}'.format(len(fun_classes), fun_classes))

    out_path = "/afs/cg.cs.tu-bs.de/home/zhang/SEDPShuffleNet/output/{}".format("munster_000168_000019_imgmsk.png")
    misc.imsave(out_path, img_msk)
    out_path = "/afs/cg.cs.tu-bs.de/home/zhang/SEDPShuffleNet/output/{}".format("munster_000168_000019_msk.png")
    misc.imsave(out_path, decoded)
    print("> Segmentation Mask Saved at: {}".format(out_path))

    msk = misc.imresize(msk, (loader.img_size[0], loader.img_size[1]))
    cv2.namedWindow("Org Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Org Mask", msk)
    cv2.namedWindow("Pre Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Pre Mask", decoded[:, :, ::-1])
    cv2.namedWindow("Image Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Image Mask", img_msk[:, :, ::-1])
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')

    parser.add_argument('--model_path', nargs='?', type=str, default='cityscapes_best_model.pkl',
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='cityscapes',
                        help='Dataset to use [\'cityscapes, mvd etc\']')
    parser.add_argument('--img_path', nargs='?', type=str, default=None,
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None,
                        help='Path of the output segmap')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    deploy_args = parser.parse_args()
    test(deploy_args)
