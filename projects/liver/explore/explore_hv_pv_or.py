import SimpleITK as sitk
import numpy as np
import os

folder = "datasets\\ircadb\\nii"

pat = ["patient-01", "patient-02", "patient-03", "patient-04"]

for p in pat:
    hv_sitk = sitk.ReadImage(os.path.join(folder, p, "mask", "hv.nii"))
    pv_sitk = sitk.ReadImage(os.path.join(folder, p, "mask", "pv.nii"))
    hv = sitk.GetArrayFromImage(hv_sitk)
    pv = sitk.GetArrayFromImage(pv_sitk)
    print("pv         : ", np.unique(pv))
    print("hv         : ", np.unique(hv))
    print("sum(hv)    : ", hv.sum())
    print("sum(pv)    : ", pv.sum())
    ves = np.logical_or(hv,pv).astype(np.uint8)
    print("hv_pv      : ", np.unique(ves))
    print("sum(hv_pv) : ", ves.sum())

    ves_sitk = sitk.GetImageFromArray(ves)
    ves_sitk.SetOrigin(hv_sitk.GetOrigin())
    ves_sitk.SetDirection(hv_sitk.GetDirection())
    ves_sitk.SetSpacing(hv_sitk.GetSpacing())
    sitk.WriteImage(ves_sitk, os.path.join(folder, p, "mask", "hv_pv.nii.gz"))
