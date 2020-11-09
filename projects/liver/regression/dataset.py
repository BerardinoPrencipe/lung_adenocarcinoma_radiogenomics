import os
import torch
import math
import numpy as np
import SimpleITK as sitk
import torch.utils.data as data_utils


class RegressionSegmentDataset(data_utils.Dataset):

    def __init__(self, parser, context, directory):

        self.parser = parser
        self.context = context
        self.directory = directory

    def __getitem__(self, item):

        livers = self.parser.getAllLivers()
        name = self.parser.getLiverNameFromIdx(item)

        liver_sitk = sitk.ReadImage(os.path.join(self.directory, name))
        _, _, thick = liver_sitk.GetSpacing()
        stride = round(6 - thick)

        liver = sitk.GetArrayFromImage(liver_sitk)
        liver = liver.astype(np.int32)

        planes = livers[name]["planes"]
        planes = np.array([coeff for el in planes.values() for coeff in el])
        planes = planes.astype(np.int32)

        # Center between left branch and right branch of the portal vein
        center = round((livers[name]["left_pv"] + livers[name]["right_pv"])/2)

        # Number of slice to take above and under the center
        n = math.ceil(self.context/2)
        low_idx = center - (n*stride)
        up_idx  = center + (n*stride)

        slices = np.arange(low_idx, up_idx, stride)

        liver = liver[slices]

        # Z-score normalization
        liver_mean = np.mean(liver, axis=0)
        liver_std = np.std(liver, axis=0)
        liver = (liver-liver_mean)/liver_std

        # Convert in torch tensor
        features = torch.from_numpy(liver).float()
        targets  = torch.from_numpy(planes).float()

        return features, targets


    def __len__(self):

        return self.parser.getLen()


# from projects.liver.geometric.JSONParser import JSONParser

# parser = JSONParser("planes.json")
# dataset = RegressionSegmentDataset(parser, 20, "D:\\Universita\\Laurea Magistrale - Computer Science Engeneering\\Tesi\\"
#                                                "LiverSegmentation\\datasets\\LiverDecathlon\\nii\\labels_segments")
# res = dataset.__getitem__(5)