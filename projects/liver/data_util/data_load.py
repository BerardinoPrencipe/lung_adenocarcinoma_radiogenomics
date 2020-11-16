import os
import torch
import numpy as np
import torch.utils.data as data_utils

from utils.utils_calc import normalize_data, normalize
from projects.liver.train.config import window_hu

DEBUG = False

# Liver Dataset - segmentation task
# when false selects both the liver and the tumor as positive labels
class LiverDataSet(torch.utils.data.Dataset):

    def __init__(self, directory, augmentation=None, context=0, do_normalize=False):

        self.augmentation = augmentation
        self.context = context
        self.directory = directory
        self.do_normalize = do_normalize
        self.data_files = os.listdir(directory)

        def get_type(s): return s[:1]
        def get_item(s): return int(s.split("_")[1].split(".")[0])
        def get_patient(s): return int(s.split("-")[1].split("_")[0])

        self.data_files.sort(key = lambda x: (get_type(x), get_patient(x), get_item(x)))
        half_dataset_len = int(len(self.data_files)/2)
        self.data_files = list(zip(self.data_files[half_dataset_len:], self.data_files[:half_dataset_len]))
    
    def __getitem__(self, idx):

        if self.context > 0:
            return load_file_context(self.data_files, idx, self.context, self.directory,
                                     self.augmentation, self.do_normalize)
        else:
            return load_file(self.data_files[idx], self.directory, self.augmentation, self.do_normalize)

    def __len__(self):

        return len(self.data_files)

    def getWeights(self):

        weights = []
        pos = 0.0
        neg = 0.0

        for data_file in self.data_files:

            _, labels = data_file
            labels = np.load(os.path.join(self.directory, labels))

            if labels.sum() > 0:
                weights.append(-1)
                pos += 1
            else:
                weights.append(0)
                neg += 1

        weights = np.array(weights).astype(float)
        weights[weights==0] = 1.0 / neg * 0.1
        weights[weights==-1] = 1.0 / pos * 0.9

        print('%d samples with positive labels, %d samples with negative labels.' % (pos, neg))

        return weights

    def getPatients(self):

        patient_dictionary = {}

        for i, data_file in enumerate(self.data_files):

            _, labels = data_file
            patient = labels.split("_")[0].split("-")[1]

            if patient in patient_dictionary:
                patient_dictionary[patient].append(i)
            else:
                patient_dictionary[patient] = [i]

        return patient_dictionary


def perform_augmentation(image, mask, augmentation):
    """ Perform Augmentation
    :param image: Image with shape C x H x W
    :param mask:  Mask  with shape 1 x H x W
    :param augmentation: imgaug object
    :return: tuple (image, mask) after augmentation
    """

    # Store shapes before augmentation to compare
    image_shape = image.shape
    mask_shape = mask.shape

    image = image.astype(np.float32)
    # Put Channels as last axis
    image = np.transpose(image, (1, 2, 0)) # H x W x C
    mask  = np.transpose(mask, (1, 2, 0))  # H x W x C

    if DEBUG:
        print("Before Augmentation")
        print("Image Shape = {}".format(image.shape))
        print("Mask  Shape = {}".format(mask.shape))
        print(f'Image dtype = {image.dtype}')

    # Albumentations augmentation
    if image.shape[2] == 1:
        image = image[:,:,0]
        mask = mask[:,:,0]
    data = {"image": image, "mask": mask}
    augmented = augmentation(**data)
    if DEBUG:
        print(f'Before augmentation - Max = {image.max()} - Min = {image.min()}')
    image, mask = augmented["image"], augmented["mask"]
    if DEBUG:
        print(f'After  augmentation - Max = {image.max()} - Min = {image.min()}')
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
    # Put Channels as first axis
    image = np.transpose(image, (2, 0, 1)) # C x H x W
    mask = np.transpose(mask, (2, 0, 1))   # C x H x W
    if DEBUG:
        print("After  Augmentation")
        print("Image Shape = {}".format(image.shape))
        print("Mask  Shape = {}".format(mask.shape))
        print(f'Image dtype = {image.dtype}')
    # Verify that shapes didn't change
    assert image.shape == image_shape, "Augmentation shouldn't change image size. Original Image Shape = {} After Augmentation = {}".format(image_shape, image.shape)
    assert mask.shape == mask_shape,   "Augmentation shouldn't change mask size.  Original Mask  Shape = {} After Augmentation = {}".format(mask_shape, mask.shape)
    return image, mask


# load data_file in directory and possibly augment
def load_file(data_file, directory, augmentation=None, do_normalize=False):

    inputs, labels = data_file
    inputs, labels = np.load(os.path.join(directory, inputs)), np.load(os.path.join(directory, labels))
    inputs, labels = np.expand_dims(inputs, 0), np.expand_dims(labels, 0)

    if augmentation is not None:
        inputs, labels = perform_augmentation(inputs, labels, augmentation=augmentation)
    if do_normalize:
        # inputs = normalize_data(inputs,interval=window_hu)
        inputs = normalize(inputs)

    labels = labels.astype(np.uint8)
    features, targets = torch.from_numpy(inputs).float(), torch.from_numpy(labels).long()
    return (features, targets)


# load data_file in directory and possibly augment including the slides above and below it
def load_file_context(data_files, idx, context, directory, augmentation=None, do_normalize=False):

    # load middle slice
    inputs_b, labels_b = data_files[idx]
    inputs_b, labels_b = np.load(os.path.join(directory, inputs_b)), np.load(os.path.join(directory, labels_b))
    inputs_b, labels_b = np.expand_dims(inputs_b, 0), np.expand_dims(labels_b, 0)

    # load slices before middle slice
    inputs_a = []
    for i in range(idx-context, idx):

        # if different patient or out of bounds, take middle slice, else load slide
        if i < 0 or data_files[idx][0][:-6] != data_files[i][0][:-6]:
            inputs = inputs_b
        else:
            inputs, _ = data_files[i]
            inputs = np.load(os.path.join(directory, inputs))
            inputs = np.expand_dims(inputs, 0)

        inputs_a.append(inputs)

    # load slices after middle slice
    inputs_c = []
    for i in range(idx+1, idx+context+1):

        # if different patient or out of bounds, take middle slice, else load slide
        if i >= len(data_files) or data_files[idx][0][:-6] != data_files[i][0][:-6]:
            inputs = inputs_b
        else:
            inputs, _ = data_files[i]
            inputs = np.load(os.path.join(directory, inputs))
            inputs = np.expand_dims(inputs, 0)

        inputs_c.append(inputs)

    # concatenate all slices for context
    # middle sice first, because the network that one for the residual connection
    inputs = [inputs_b] + inputs_a + inputs_c
    labels = labels_b
    labels = labels.astype(np.uint8)

    inputs = np.concatenate(inputs, 0)

    if augmentation is not None:
        inputs, labels = perform_augmentation(inputs, labels, augmentation=augmentation)
    if do_normalize:
        # inputs = normalize_data(inputs,interval=window_hu)
        inputs = normalize(inputs)

    features, targets = torch.from_numpy(inputs).float(), torch.from_numpy(labels).long()
    return (features, targets)
