import torch
import numpy as np
import os
import cv2
from utils import imresizeNoStretch, normalize


class DataGen(torch.utils.data.Dataset):

    def __init__(self, images, masks,
                 batch_size=1, image_size=(1024,1024), channels_size=3, n_outs = 1,
                 augmentation=None, loader=None):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels_size = channels_size
        self.augmentation = augmentation
        if loader is not None:
            self.load_function = loader
        else:
            self.load_function = self.loader
        self.n_outs = n_outs
        self.on_epoch_end()

    def __load__(self, image_path, mask_path):

        image, mask = self.load_function(image_path, mask_path)
        image = self.normalize_image(image)
        image, mask = self.adapt_to_size(image, mask)
        image, mask = self.crop_image(image, mask)

        if self.augmentation is not None:
            image, mask = self.perform_augmentation(image, mask)

        # Cast label and mask dtypes
        image, mask = image.astype(np.float32), mask.astype(np.float32)

        # if self.n_outs > 1:
        #    mask = self.one_hot_encode_mask(mask)
        # else:
        #   mask = mask.astype(np.bool)

        image, mask = np.transpose(image, (2,0,1)), mask[:,:,0]

        return image, mask

    def __getitem__(self, index):
        files_batch, masks_batch = self.images[index], self.masks[index]
        scan_id, mask_id = files_batch, masks_batch
        image, mask = self.__load__(scan_id, mask_id)
        image, mask = torch.from_numpy(image).float(), torch.from_numpy(mask).float()
        return image, mask

    def on_epoch_end(self):
        pass

    def normalize_image(self, image):
        return normalize(image)

    def loader(self, image_path, mask_path):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        return image, mask

    def adapt_to_size(self, image, mask):
        image = imresizeNoStretch(image, self.image_size[0], interpolation=cv2.INTER_LINEAR)
        mask = imresizeNoStretch(mask, self.image_size[0], interpolation=cv2.INTER_NEAREST)
        return image, mask

    def crop_image(self, image, mask):
        return image, mask

    def one_hot_encode_mask(self, mask):
        label = np.zeros((mask.shape[0], mask.shape[1], self.n_outs))
        # One-hot encode
        for out in range(0, self.n_outs):
            one_hot = [0] * self.n_outs
            one_hot[out] = 1
            label[mask[:, :, 0] == out] = one_hot
        return label

    def perform_augmentation(self, image, mask):
        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = self.augmentation.to_deterministic()
        image = det.augment_image(image.astype(np.float32))
        mask = det.augment_image(mask.astype(np.uint8))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        return image, mask

    def __len__(self):
        return int(np.ceil(len(self.images) / float(self.batch_size)))


def get_balancing_weights(train_labels, classes):
    n_classes = len(classes)
    tot_for_class = np.zeros(n_classes)
    tot_tot = 0
    weights = np.zeros(n_classes)
    for train_label_path in train_labels:
        label = cv2.imread(train_label_path)
        for i in range(0, n_classes):
            n_class_i = np.count_nonzero(label==i)
            tot_for_class[i] += n_class_i
            tot_tot += n_class_i

    for i in range(0,n_classes):
        weights[i] = np.log(tot_tot/tot_for_class[i])

    return weights