import os
import torch
import scipy.misc as misc

from torch.utils import data
from datasets.augmentations import *
from scripts.utils import recursive_glob


class CityscapesLoader(data.Dataset):
    """
    CityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]]

    label_colours = dict(zip(range(19), colors))

    def __init__(self, root, split="train", gt="gtCoarse", img_size=(512, 1024),
                 is_transform=False, augmentations=None):
        """
        :param root:         (str)  Path to the data sets root
        :param split:        (str)  Data set split -- 'train' 'train_extra' or 'val'
        :param gt:           (str)  Type of ground truth label -- 'gtFine' or 'gtCoarse'
        :param img_size:     (tuple or int) The size of the input image
        :param is_transform: (bool) Transform the image or not
        :param augmentations (object) Data augmentations used in the image and label
        """
        self.root = root
        self.gt = gt
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations

        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([73.16, 82.91, 72.39])
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, gt, self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                            'motorcycle', 'bicycle']

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("> No files for split=[%s] found in %s" % (split, self.images_base))

        print("> Found %d %s images..." % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + '{}_labelIds.png'.format(self.gt))

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))

        img = misc.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        # img = misc.imresize(img, (self.img_size[0], self.img_size[1], "bilinear"))

        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))

        lbl = misc.imread(lbl_path)
        # lbl = misc.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode='F')
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = img[:, :, ::-1]         # From RGB to BGR
        img = img.astype(float)
        img -= self.mean
        img /= 255.0
        img = img.transpose(2, 0, 1)  # From H*W*C to C*H*W

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            raise ValueError("> Segmentation map contained invalid class values.")

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


# +++++++++++++++++++++++++++++++++++++++++++++ #
# Test the code of 'CityscapesLoader'
# +++++++++++++++++++++++++++++++++++++++++++++ #
if __name__ == '__main__':
    net_h, net_w = 448, 896
    augment = Compose([RandomHorizontallyFlip(), RandomSized((0.625, 0.75)),
                       RandomRotate(6), RandomCrop((net_h, net_w))])

    local_path = "/zfs/zhang/Cityscapes"
    dst = CityscapesLoader(local_path, split="train_extra", is_transform=True, augmentations=augment)

    """
    color_map = dst.label_colours
    class_names = dst.class_names

    grid_height = int(net_h//dst.n_classes)
    start_pixel = int((net_h % dst.n_classes) / 2)

    color_bar = np.ones((net_h, 120, 3), dtype=np.uint8)
    for label_id in np.arange(dst.n_classes):
        end_pixel = start_pixel + grid_height
        color_bar[start_pixel:end_pixel, :, :] = color_map[label_id]

        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(color_bar, class_names[label_id+1],
                    (2, start_pixel + 5 + int(grid_height//2)),
                    font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        start_pixel = end_pixel

    cv2.namedWindow("color bar", cv2.WINDOW_NORMAL)
    cv2.imshow("color bar", color_bar)
    cv2.waitKey(0)
    """

    bs = 6
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=2, shuffle=True)
    for i, data in enumerate(trainloader):
        print("batch :", i)
        """
        imgs, labels = data
        imgs = imgs.numpy()[:, :, :, ::-1]
        labels = labels.numpy()

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", np.squeeze(imgs.astype(np.uint8)))

        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Mask", np.squeeze(labels))
        cv2.waitKey(0)
        """