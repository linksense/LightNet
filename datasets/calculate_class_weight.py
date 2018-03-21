import numpy as np
import os

from scripts.utils import recursive_glob


def calc_median_frequency(classes, present_num):
    """
    Class balancing by median frequency balancing method.
    Reference: https://arxiv.org/pdf/1411.4734.pdf
       'a = median_freq / freq(c) where freq(c) is the number of pixels
        of class c divided by the total number of pixels in images where
        c is present, and median_freq is the median of these frequencies.'
    """
    class_freq = classes / present_num
    median_freq = np.median(class_freq)
    return median_freq / class_freq


def calc_log_frequency(classes, value=1.02):
    """Class balancing by ERFNet method.
       prob = each_sum_pixel / each_sum_pixel.max()
       a = 1 / (log(1.02 + prob)).
    """
    class_freq = classes / classes.sum()  # ERFNet is max, but ERFNet is sum
    # print(class_freq)
    # print(np.log(value + class_freq))
    return 1 / np.log(value + class_freq)


if __name__ == '__main__':
    import os
    import scipy.misc as misc
    from datasets.cityscapes_loader import CityscapesLoader

    method = "median"
    result_path = "/afs/cg.cs.tu-bs.de/home/zhang/SEDPShuffleNet/datasets"

    traval = "gtFine"
    imgs_path = "/zfs/zhang/Cityscapes/leftImg8bit/train"
    lbls_path = "/zfs/zhang/Cityscapes/gtFine/train"
    images = recursive_glob(rootdir=imgs_path, suffix='.png')

    num_classes = 19

    local_path = "/zfs/zhang/Cityscapes"
    dst = CityscapesLoader(local_path, gt="gtFine", split="train", is_transform=True, augmentations=None)

    classes, present_num = ([0 for i in range(num_classes)] for i in range(2))

    for idx, img_path in enumerate(images):
        lbl_path = os.path.join(lbls_path, img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + '{}_labelIds.png'.format(traval))

        lbl = misc.imread(lbl_path)
        lbl = dst.encode_segmap(np.array(lbl, dtype=np.uint8))

        for nc in range(num_classes):
            num_pixel = (lbl == nc).sum()
            if num_pixel:
                classes[nc] += num_pixel
                present_num[nc] += 1

    if 0 in classes:
        raise Exception("Some classes are not found")

    classes = np.array(classes, dtype="f")
    presetn_num = np.array(classes, dtype="f")
    if method == "median":
        class_weight = calc_median_frequency(classes, present_num)
    elif method == "log":
        class_weight = calc_log_frequency(classes)
    else:
        raise Exception("Please assign method to 'mean' or 'log'")

    print("class weight", class_weight)
    result_path = os.path.join(result_path, "{}_class_weight.npy".format(method))
    np.save(result_path, class_weight)
    print("Done!")
