import os
import pandas as pd

dataset_folder = './feature_extraction/NSCLC_nifti'
dataset_folder = os.path.abspath(dataset_folder)
image_folder = os.path.join(dataset_folder, "images")
masks_folder = os.path.join(dataset_folder, "masks")

file_dict = {
    "Image": [],
    "Mask": []
}

imgs_file = os.listdir(image_folder)

for file in imgs_file:

    file_dict["Image"].append(os.path.join(image_folder, file))
    file_dict["Mask"].append(os.path.join(masks_folder, file))

file_pd = pd.DataFrame(file_dict)
file_pd.to_csv("./feature_extraction/dataset_NSCLC.csv")
