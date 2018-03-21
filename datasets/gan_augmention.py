import os
import imageio
from shutil import copyfile


if __name__ == "__main__":
    root_dir = "/afs/cg.cs.tu-bs.de/home/zhang/PycharmProjects/pix2pixHD/results/label2city_1024p/test_latest"
    gan_dir = os.path.join(root_dir, "images")
    gan_label_dir = "/afs/cg.cs.tu-bs.de/home/zhang/PycharmProjects/pix2pixHD/datasets/cityscapes/test_label"

    image_dir = os.path.join(root_dir, "leftImg8bit", "gan")
    label_dir = os.path.join(root_dir, "gtFine", "gan")

    all_files = os.listdir(gan_dir)
    all_files.sort()

    synthesized_images = list(filter(lambda x: '_synthesized_image.jpg' in x, all_files))
    labels = os.listdir(gan_label_dir)
    labels.sort()

    for index, names in enumerate(zip(synthesized_images, labels)):
        image_name, label_name = names[0], names[1]

        image = imageio.imread(os.path.join(gan_dir, image_name))
        image_name = image_name.replace("_gtFine_labelIds_synthesized_image.jpg", "_leftImg8bit.png")

        name_split = image_name.split("_")
        city_name = str(name_split[0])
        iid1 = str(name_split[1])
        iid2 = str(name_split[2])
        new_iid2 = "{}".format(str(index).zfill(6))

        image_name = image_name.replace(city_name, "gan")
        image_name = image_name.replace(iid1, "000000")
        image_name = image_name.replace(iid2, new_iid2)

        print("> Processing Image: {}".format(image_name))
        imageio.imwrite(os.path.join(image_dir, image_name), image)

        new_label_name = label_name.replace(city_name, "gan")
        new_label_name = new_label_name.replace(iid1, "000000")
        new_label_name = new_label_name.replace(iid2, new_iid2)

        copyfile(os.path.join(gan_label_dir, label_name), os.path.join(label_dir, new_label_name))

        if (os.path.basename(image_name)[:-15] + 'gtFine_labelIds.png') != new_label_name:
            raise Exception("> Name mismatch !!!")

    print("> ++++++++++++++++++++++++++++++++++++++++++++ <")
    print("> Done!!!")
    print("> ++++++++++++++++++++++++++++++++++++++++++++ <")
