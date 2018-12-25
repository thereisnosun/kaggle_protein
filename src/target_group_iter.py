import numpy as np
from PIL import Image
from scipy.misc import imread


def load_image(basepath, image_id):
    images = np.zeros(shape=(4,512,512))
    images[0,:,:] = imread(basepath + image_id + "_green" + ".png")
    images[1,:,:] = imread(basepath + image_id + "_red" + ".png")
    images[2,:,:] = imread(basepath + image_id + "_blue" + ".png")
    images[3,:,:] = imread(basepath + image_id + "_yellow" + ".png")
    return images


def make_image_row(image, subax, title):
    subax[0].imshow(image[0], cmap="Greens")
    subax[1].imshow(image[1], cmap="Reds")
    subax[1].set_title("stained microtubules")
    subax[2].imshow(image[2], cmap="Blues")
    subax[2].set_title("stained nucleus")
    subax[3].imshow(image[3], cmap="Oranges")
    subax[3].set_title("stained endoplasmatic reticulum")
    subax[0].set_title(title)
    return subax


class TargetGroupIterator:

    def __init__(self, reverse_train_labels, train_labels, target_names, batch_size, basepath):
        self.target_names = target_names
        self.target_list = [reverse_train_labels[key] for key in target_names]
        self.batch_shape = (batch_size, 4, 512, 512)
        self.basepath = basepath
        self.train_labels = train_labels

    def find_matching_data_entries(self):
        self.train_labels["check_col"] = self.train_labels.Target.apply(
            lambda l: self.check_subset(l)
        )
        self.images_identifier = self.train_labels[self.train_labels.check_col == 1].Id.values
        self.train_labels.drop("check_col", axis=1, inplace=True)

    def check_subset(self, targets):
        return np.where(set(targets).issubset(set(self.target_list)), 1, 0)

    def get_loader(self):
        filenames = []
        idx = 0
        images = np.zeros(self.batch_shape)
        for image_id in self.images_identifier:
            images[idx, :, :, :] = load_image(self.basepath, image_id)
            filenames.append(image_id)
            idx += 1
            if idx == self.batch_shape[0]:
                yield filenames, images
                filenames = []
                images = np.zeros(self.batch_shape)
                idx = 0
        if idx > 0:
            yield filenames, images

    def make_title(self, file_id, label_names):
        file_targets = self.train_labels.loc[self.train_labels.Id == file_id, "Target"].values[0]
        title = " - "
        for n in file_targets:
            title += label_names[n] + " - "
        return title