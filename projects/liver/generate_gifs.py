import os
from utils import create_gif

gifs_folder = 'gifs'
masmec_dataset_path = 'datasets/Masmec'
subfolders = os.listdir(masmec_dataset_path)
subfolders_paths = [os.path.join(masmec_dataset_path, subfolder) for subfolder in subfolders]
masks_paths  = list()
images_paths = list()
for subfolder_path in subfolders_paths:
    subfolder_list = os.listdir(subfolder_path)
    mask_path  = os.path.join(subfolder_path, next(path for path in subfolder_list if path[:12]=="segmentation"))
    image_path = os.path.join(subfolder_path, next(path for path in subfolder_list if path[:6]=="volume"))
    masks_paths.append(mask_path)
    images_paths.append(image_path)

for idx, (image_path, mask_path) in enumerate(zip(images_paths, masks_paths)):
    path_out = os.path.join(gifs_folder, "segm_{:04d}.gif".format(idx))
    create_gif(image_path, mask_path, path_out)