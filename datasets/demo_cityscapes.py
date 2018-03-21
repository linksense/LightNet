import os
import time

import cv2
import numpy as np
import scipy.misc as misc
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from datasets.cityscapes_loader import CityscapesLoader
from models.mobilenetv2plus import MobileNetV2Plus
from models.sewrnetv2 import SEWiderResNetV2
from modules import InPlaceABNWrapper
from functools import partial


def test(video_root, output_root, model_path):
    net_h, net_w, color_bar_w = 896, 1792, 120
    frame_size = (net_w + color_bar_w, net_h)
    codec = cv2.VideoWriter_fourcc(*'MJPG')

    data_path = "/zfs/zhang/Cityscapes"
    loader = CityscapesLoader(data_path, is_transform=True, split='val',
                              img_size=(net_h, net_w), augmentations=None)
    n_classes = loader.n_classes

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 0. Setup Color Bar
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    color_map = loader.label_colours
    class_names = loader.class_names

    grid_height = int(net_h // loader.n_classes)
    start_pixel = int((net_h % loader.n_classes) / 2)

    color_bar = np.ones((net_h, color_bar_w, 3), dtype=np.uint8)*128
    for label_id in np.arange(loader.n_classes):
        end_pixel = start_pixel + grid_height
        color_bar[start_pixel:end_pixel, :, :] = color_map[label_id]

        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(color_bar, class_names[label_id + 1],
                    (2, start_pixel + 5 + int(grid_height // 2)),
                    font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        start_pixel = end_pixel
    color_bar = color_bar[:, :, ::-1]

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 1. Setup Model
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("> 1. Setting up Model...")
    model = MobileNetV2Plus(n_class=n_classes, in_size=(net_h, net_w), width_mult=1.0,
                            out_sec=256, aspp_sec=(12*2, 24*2, 36*2),
                            norm_act=partial(InPlaceABNWrapper, activation="leaky_relu", slope=0.1))

    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    # state = convert_state_dict(torch.load("/media/datavolume3/huijun/SEDPShuffleNet/weights/{}".format(
    #     args.model_path))['model_state'])
    pre_weight = torch.load(model_path)['model_state']
    model.load_state_dict(pre_weight)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 2. Inference Model
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    org_video_sub = os.listdir(video_root)
    org_video_sub.sort()
    prd_video_sub = os.listdir(output_root)
    prd_video_sub.sort()

    my_writer = cv2.VideoWriter("Cityscapes_Result.avi", codec, 24.0, frame_size)
    for v_id in np.arange(len(org_video_sub)):
        assert org_video_sub[v_id] == prd_video_sub[v_id]
        print("> 2. Processing Video # {}...".format(v_id))
        curr_video_path = os.path.join(video_path, org_video_sub[v_id])
        images_name = os.listdir(curr_video_path)
        images_name.sort()

        for img_id in np.arange(len(images_name)):
            curr_image = images_name[img_id]
            print("> Processing Video #{} Image: {}...".format(v_id, curr_image))

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 2.1 Pre-processing Image
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            curr_img_path = os.path.join(curr_video_path, curr_image)
            image = misc.imread(curr_img_path)
            image = np.array(image, dtype=np.uint8)

            start_time = time.time()
            resized_img = misc.imresize(image, (loader.img_size[0], loader.img_size[1]), interp='bilinear')
            image = misc.imresize(image, (loader.img_size[0], loader.img_size[1]), interp='bilinear')

            image = image[:, :, ::-1]         # RGB -> BGR
            image = image.astype(float)
            image -= loader.mean
            image /= 255.0

            image = image.transpose(2, 0, 1)  # HWC -> CHW
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image).float()

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 2.2 Prediction/Inference
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            model.eval()
            images = Variable(image.cuda(), volatile=True)

            outputs = F.softmax(model(images), dim=1)
            pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)

            decoded = loader.decode_segmap(pred) * 255
            decoded = decoded.astype(np.uint8)
            print("> Processed Video #{} Image: {}, Time: {}s".format(v_id, curr_image, (time.time() - start_time)))

            img_msk = cv2.addWeighted(resized_img, 0.55, decoded, 0.45, 0)
            img_msk = img_msk[:, :, ::-1]  # RGB

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 2.3 Saving prediction result
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
            save_path = os.path.join(output_root, prd_video_sub[v_id], curr_image)
            cv2.imwrite(save_path, img_msk)

            # img_msk_color = np.zeros((net_h, net_w + 120, 3))
            img_msk_color = np.concatenate((img_msk, color_bar), axis=1)

            # cv2.imshow("show", img_msk_color)
            # cv2.waitKey(0)
            my_writer.write(img_msk_color)

    print("> # +++++++++++++++++++++++++++++++++++++++ #")
    print("> Done!!!")
    print("> # +++++++++++++++++++++++++++++++++++++++ #")
    my_writer.release()


if __name__ == '__main__':
    video_path = "/zfs/zhang/Cityscapes/leftImg8bit/demoVideo"
    output_path = "/zfs/zhang/Cityscapes/demo"
    model_weight = "/zfs/zhang/TrainLog/weights/{}".format("cityscapes_mobilenetv2_best_model.pkl")
    test(video_path, output_path, model_weight)
