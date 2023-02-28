import os
import pydicom
import pandas as pd

dataset_folder = '.\\feature_extraction\\NSCLC_nifti'
dataset_folder = os.path.abspath(dataset_folder)
patients = os.listdir(dataset_folder)

file_dict = {
    "Image": [],
    "Mask": []
}

for patient in patients:

    pat_files_path = os.path.join(dataset_folder, patient)
    pat_files = os.listdir(pat_files_path)

    for pat_file in pat_files:

        pat_file_split = pat_file.split(".")[0]
        path = os.path.join(pat_files_path, pat_file)

        if pat_file_split.endswith("mask"):

            file_dict["Mask"].append(path)

        if pat_file_split.endswith("scan"):

            file_dict["Image"].append(path)

file_pd = pd.DataFrame(file_dict)
file_pd.to_csv(".\\feature_extraction\\dataset_NSCLC.csv")
