import torch
import json
import os

from scripts.utils import recursive_glob
from datasets.augmentations import *
from torch.utils import data
from PIL import Image


class MapillaryVistasLoader(data.Dataset):
    """
    MapillaryVistasLoader
    https://www.mapillary.com/dataset/vistas
    https://blog.mapillary.com/product/2017/05/03/mapillary-vistas-dataset.html

    Data is derived from Mapillary, and can be downloaded from here (need request):
    https://www.mapillary.com/dataset/vistas
    """

    def __init__(self, root, split="training", img_size=(640, 1280), is_transform=True, augmentations=None):
        """
        :param root:         (str)  Path to the data sets root
        :param split:        (str)  Data set split -- 'training' or 'validation'
        :param img_size:     (tuple or int) The size of the input image
        :param is_transform: (bool) Transform the image or not
        :param augmentations (object) Data augmentations used in the image and label
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 65

        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([80.5423, 91.3162, 81.4312])
        self.files = {}

        self.images_base = os.path.join(self.root, self.split, 'images')
        self.annotations_base = os.path.join(self.root, self.split, 'labels')

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.jpg')

        self.class_ids, self.class_names, self.class_colors = self._parse_config()

        self.ignore_id = 65

        if not self.files[split]:
            raise Exception("> No files for split=[%s] found in %s" % (split, self.images_base))

        print("> Found %d %s images..." % (len(self.files[split]), split))

    @staticmethod
    def _parse_config():
        # read in config file
        with open('/afs/cg.cs.tu-bs.de/home/zhang/SEDPShuffleNet/datasets/config.json') as config_file:
            config = json.load(config_file)

        labels = config['labels']

        class_names = []
        class_ids = []
        class_colors = []
        print("> There are {} labels in the config file".format(len(labels)))
        for label_id, label in enumerate(labels):
            class_names.append(label["readable"])
            class_ids.append(label_id)
            class_colors.append(label["color"])

        return class_names, class_ids, class_colors

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base, os.path.basename(img_path).replace(".jpg", ".png"))

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))

        img = Image.open(img_path)
        img = np.array(img, dtype=np.uint8)
        # print("> Image size: {}, {}".format(img.shape[0], img.shape[1]))

        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))

        lbl = Image.open(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)

        # classes = np.unique(lbl)
        # print("> Number of classes before resize: {}".format(classes))
        # lbl = misc.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode='F')
        # print("> Number of classes after resize: {}".format(classes))
        # if not np.all(classes == np.unique(lbl)):
        #     print("> !!!!!!!!!!!!!!!! WARN: resizing labels yielded fewer classes. !!!!!!!!!!!!!!!!!!!!")

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
        img = img.transpose(2, 0, 1)  # From HWC to CHW

        return img, lbl

    def apply_color_map(self, image_array):
        color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

        for cls_id, color in enumerate(self.class_colors):
            # set all pixels with the current label to the color of the current label
            color_array[image_array == cls_id] = color

        return color_array


# +++++++++++++++++++++++++++++++++++++++++++++ #
# Test the code of 'MapillaryVistasLoader'
# +++++++++++++++++++++++++++++++++++++++++++++ #
if __name__ == '__main__':
    net_h, net_w = 448, 896
    augment = Compose([RandomHorizontallyFlip(), RandomSized((0.625, 0.75)),
                       RandomRotate(6), RandomCrop((net_h, net_w))])

    local_path = '/zfs/zhang/MVD'
    dst = MapillaryVistasLoader(local_path, img_size=(net_h, net_w), is_transform=True, augmentations=augment)
    bs = 8
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=4, shuffle=True)
    for i, data in enumerate(trainloader):
        print("batch :", i)
        """
        imgs, labels = data
        imgs = imgs.numpy()[:, :, :, ::-1]

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", np.squeeze(imgs.astype(np.uint8)))

        # cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        # cv2.imshow("Mask", labels[0])
        cv2.waitKey(0)
        """