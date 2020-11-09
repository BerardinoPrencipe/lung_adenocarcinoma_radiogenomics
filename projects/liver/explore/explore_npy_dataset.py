import os
import numpy as np
import matplotlib.pyplot as plt

# path_to_npy_dataset = 'datasets/LiverDecathlon/npy_vessels_masked/train'
path_to_npy_dataset = 'datasets/LiverDecathlon/npy_masked/train'
z_size = 41
arr_3d = np.zeros((z_size, 512, 512))
image_paths = []
for idx in range(0,z_size):
    image_path = os.path.join(path_to_npy_dataset, 'segmentation-050_{}.npy'.format(idx))
    image_paths.append(image_path)
    arr_3d[idx, :, :] = np.load(image_path)

