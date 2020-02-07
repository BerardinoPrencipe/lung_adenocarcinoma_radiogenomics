import numpy as np
import torch
import time
import SimpleITK as sitk
import nibabel as nib
import os
import cv2
import imageio
import matplotlib

from utils_calc import normalize_data, normalize

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

cuda = torch.cuda.is_available()

def print_dict(dict_to_print):
    for x in dict_to_print:
        print("{:15s} : {}".format(x, dict_to_print[x]))


# returns the patient number from the filename


def imresizeNoStretch(image, desired_size=1024,
                      color=None, zeroPadding=True, return_also_dim_pad=False, interpolation=cv2.INTER_LINEAR):
    old_size = image.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    image = cv2.resize(image,(new_size[1], new_size[0]),interpolation=interpolation)
    if len(image.shape) == 2:
        image = np.expand_dims(image,axis=2)
    if zeroPadding:
        new_im = zeroPad(image,[desired_size,desired_size],color)
    else:
        new_im = image
    if return_also_dim_pad:
        return new_im, new_size
    else:
        return new_im


def zeroPad(image,desired_size,color=None):

    des_height,des_width = desired_size
    height,width,channels = image.shape
    delta_w = des_width - width
    delta_h = des_height - height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    if color is None:
        color = list(np.zeros(channels))

    new_im = cv2.copyMakeBorder(image, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    if len(new_im.shape)==2:
        new_im = np.expand_dims(new_im,axis=2)
    return new_im


def montage(imgs, dim=None, consistency=True, normalization=True):
    if dim is None:
        dim = (1, 1)
    nrows, ncols = dim

    dims = imgs[0].shape
    if consistency:
        for img in imgs:
            assert img.shape == dims, 'Mismatch in dimensions'

    if normalization:
        imgs = [normalize(img) for img in imgs]

    initialize_zeros = (dims[0] * nrows, dims[1] * ncols) if len(dims)==2 else (dims[0] * nrows, dims[1] * ncols, dims[2])
    img_out = np.zeros(initialize_zeros)

    for i in range(0, nrows):
        for j in range(0, ncols):
            img_out[i * dims[0]:(i + 1) * dims[0], j * dims[1]:(j + 1) * dims[1]] = imgs[i * ncols + j]

    return img_out


def labeloverlay(image, label, classes=None, colors=None):
    assert len(classes) == len(colors), "Mismatch between number of classes and number of colors"
    for idx, color in enumerate(colors):
        mask = np.zeros(label.shape,dtype=np.bool)
        mask[label==idx] = 1
        image = apply_mask(image, mask, color=color)
    return image


def apply_mask(image, mask_2d, transparency=0.25, color=None, show_results=False):
    """ Apply a 2D binary mask_healthy to image.

    :param image: original image
    :param mask_2d: 2d binary mask_healthy
    :param transparency: degree of transparency of mask_healthy when overlayed with image
    :param color: color of mask_healthy when overlayed with image. Green = (0,1,0). Red = (1,0,0).
    :param show_results: plot results
    :return: output image, which is composed by original image with mask_2d overlayed
    """

    if image.dtype == 'uint8':
        #convert to floating point
        img = np.array(image, dtype=np.float)
        img /= 255.0
    else:
        img = image.copy()

    if show_results:
        plt.imshow(img)
        plt.show()

    mask_3d = np.zeros(image.shape)
    n_channels = image.shape[2]

    for i in range(0,n_channels):
        mask_3d[:,:,i] = mask_2d.copy()
    #convert to floating point
    mask = np.array(mask_3d, dtype=np.float)

    mask*=transparency

    if show_results:
        plt.imshow(mask)
        plt.show()


    if color is None:
        # make a green overlay as default
        color = (0,1,0)

    color_overlay = np.ones(img.shape, dtype=np.float)*color

    #color over original image
    out_image = color_overlay*mask + img*(1.0-mask)

    if show_results:
        plt.imshow(out_image)
        plt.show()

    return out_image


def imshow_components(labels):
    # Map component labels to hue value
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # set bg label to black
    labeled_img[label_hue==0] = 0

    plt.imshow(labeled_img)
    plt.show()


def create_gif(image_path, mask_path, path_out):
    images = []
    image3d, mask3d = nib.load(image_path), nib.load(mask_path)
    image3d, mask3d = image3d.get_data(), mask3d.get_data()
    image3d, mask3d = np.transpose(image3d, (1, 0, 2)), np.transpose(mask3d, (1, 0, 2))
    num_slices = image3d.shape[2]
    image3d = normalize_data(image3d, -200, 200)
    image3d, mask3d = image3d * 128, mask3d * 127
    for i in reversed(range(num_slices)):
        image_montage = montage([image3d[:,:,i],image3d[:,:,i]+mask3d[:,:,i]],dim=(1,2),normalization=False)
        image_montage = image_montage.astype(np.uint8)
        images.append(image_montage)
    imageio.mimsave(path_out, images, format='GIF')