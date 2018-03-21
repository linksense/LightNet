"""
    This file shows how to load and use the dataset
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def apply_color_map(image_array, labels):
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    for label_id, label in enumerate(labels):
        # set all pixels with the current label to the color of the current label
        color_array[image_array == label_id] = label["color"]

    return color_array


def main(mvd_dir):
    # a nice example
    key = 'M2kh294N9c72sICO990Uew'

    # read in config file
    with open('config.json') as config_file:
        config = json.load(config_file)
    # in this example we are only interested in the labels
    labels = config['labels']

    # print labels
    print("There are {} labels in the config file".format(len(labels)))
    for label_id, label in enumerate(labels):
        print("{:>30} ({:2d}): {:<50} has instances: {}".format(label["readable"], label_id, label["name"], label["instances"]))

    # set up paths for every image
    image_path = "{}training/images/{}.jpg".format(mvd_dir, key)
    label_path = "{}training/labels/{}.png".format(mvd_dir, key)
    instance_path = "{}training/instances/{}.png".format(mvd_dir, key)

    # load images
    base_image = Image.open(image_path)
    label_image = Image.open(label_path)
    instance_image = Image.open(instance_path)

    # convert labeled data to numpy arrays for better handling
    label_array = np.array(label_image)
    instance_array = np.array(instance_image, dtype=np.uint16)

    # now we split the instance_array into labels and instance ids
    instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
    instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)

    # for visualization, we apply the colors stored in the config
    colored_label_array = apply_color_map(label_array, labels)
    colored_instance_label_array = apply_color_map(instance_label_array, labels)

    # plot the result
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,15))

    ax[0][0].imshow(base_image)
    ax[0][0].get_xaxis().set_visible(False)
    ax[0][0].get_yaxis().set_visible(False)
    ax[0][0].set_title("Base image")
    ax[0][1].imshow(colored_label_array)
    ax[0][1].get_xaxis().set_visible(False)
    ax[0][1].get_yaxis().set_visible(False)
    ax[0][1].set_title("Labels")
    ax[1][0].imshow(instance_ids_array)
    ax[1][0].get_xaxis().set_visible(False)
    ax[1][0].get_yaxis().set_visible(False)
    ax[1][0].set_title("Instance IDs")
    ax[1][1].imshow(colored_instance_label_array)
    ax[1][1].get_xaxis().set_visible(False)
    ax[1][1].get_yaxis().set_visible(False)
    ax[1][1].set_title("Labels from instance file (identical to labels above)")

    fig.savefig('MVD_plot.png')


if __name__ == '__main__':
    mvd_root = "/zfs/zhang/MVD/"
    main(mvd_root)
