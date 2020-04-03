import cv2
import json
import os
import pickle as pk
import random
import numpy as np
import SimpleITK as sitk
from projects.liver.geometric.points_segments import point_to_segments, find_lines, find_lines_one, \
                                                        extract_segments, plane_to_lines, extract_contours, calculate_coefficent, \
                                                        extract_segments_plane, evaluate_segments

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

liver_folder   = "D:\\Universita\\Laurea Magistrale - Computer Science Engeneering\\Tesi\\LiverSegmentation\\datasets\\LiverDecathlon\\nii\\labels_liver\\"
dataset_folder = "D:\\Universita\\Laurea Magistrale - Computer Science Engeneering\\Tesi\\LiverSegmentation\\datasets/LiverDecathlon/nii/labels_segments"
result_folder  = "D:\\Universita\\Laurea Magistrale - Computer Science Engeneering\\Tesi\\LiverSegmentation\\datasets\\LiverDecathlon\\nii\\rebuilt_plane_label"
files_name = os.listdir(dataset_folder)

files_name.remove('hepaticvessel_005.nii.gz')
files_name.remove('hepaticvessel_063.nii.gz')
files_name.remove('hepaticvessel_129.nii.gz')
files_name.remove('hepaticvessel_133.nii.gz')
files_name.remove('hepaticvessel_159.nii.gz')
files_name.remove('hepaticvessel_237.nii.gz')
files_name.remove('hepaticvessel_347.nii.gz')

results = {}
for name in files_name:
    liver_seg_sitk = sitk.ReadImage(os.path.join(dataset_folder,name))
    liver_seg = sitk.GetArrayFromImage(liver_seg_sitk)
    liver_seg = liver_seg.astype(np.uint8)

    liver_sitk = sitk.ReadImage(os.path.join(liver_folder,name))
    liver = sitk.GetArrayFromImage(liver_sitk)
    liver = liver.astype(np.uint8)

    res = find_lines(liver_seg)
    res['planes']['2,4'] = tuple([float(i) for i in res['planes']['2,4']['plane']])
    res['planes']['4,5'] = tuple([float(i) for i in res['planes']['4,5']['plane']])
    res['planes']['7,8'] = tuple([float(i) for i in res['planes']['7,8']['plane']])
    res['left_pv'] = int(res['left_pv'])
    res['right_pv'] = int(res['right_pv'])
    print(name)
    results[name] = res

    r = extract_segments_plane(res, liver)
    rebuilt_sitk = sitk.GetImageFromArray(r)
    rebuilt_sitk.SetOrigin(liver_sitk.GetOrigin())
    rebuilt_sitk.SetSpacing(liver_sitk.GetSpacing())
    rebuilt_sitk.SetDirection(liver_sitk.GetDirection())
    sitk.WriteImage(rebuilt_sitk, os.path.join(result_folder, name))

DSCs = []
for name in files_name:
    gt_sitk = sitk.ReadImage(os.path.join(dataset_folder,name))
    gt = sitk.GetArrayFromImage(gt_sitk)
    pred_sitk = sitk.ReadImage(os.path.join(result_folder,name))
    pred = sitk.GetArrayFromImage(pred_sitk)
    DSCs.append(evaluate_segments(gt,pred))
    print(name)


check = [5, 7, 10, 29, 30, 31,38,42,45,50,56,57,61,66,67,71,82,88,
         91,109,111,112,117,118,122,123,125,133,140,147,150,
         172,179,181]
DSCs_copy = DSCs.copy()

for el in check:
    DSCs_copy.remove(DSCs[el])


seg2 = [a[2] for a in DSCs_copy]
seg3 = [a[3] for a in DSCs_copy]
seg4 = [a[4] for a in DSCs_copy]
seg5 = [a[5] for a in DSCs_copy]
seg6 = [a[6] for a in DSCs_copy]
seg7 = [a[7] for a in DSCs_copy]
seg8 = [a[8] for a in DSCs_copy]

print("seg 2 mean: ", np.mean(np.array(seg2)))
print("seg 2 std:  ", np.mean(np.std(seg2)))
print("seg 2 max:  ", np.mean(np.max(seg2)))
print("seg 2 min:  ", np.mean(np.min(seg2)))
print(" ")
print("seg 3 mean: ", np.mean(np.array(seg3)))
print("seg 3 std:  ", np.mean(np.std(seg3)))
print("seg 3 max:  ", np.mean(np.max(seg3)))
print("seg 3 min:  ", np.mean(np.min(seg3)))
print(" ")
print("seg 4 mean: ", np.mean(np.array(seg4)))
print("seg 4 std:  ", np.mean(np.std(seg4)))
print("seg 4 max:  ", np.mean(np.max(seg4)))
print("seg 4 min:  ", np.mean(np.min(seg4)))
print(" ")
print("seg 5 mean: ", np.mean(np.array(seg5)))
print("seg 5 std:  ", np.mean(np.std(seg5)))
print("seg 5 max:  ", np.mean(np.max(seg5)))
print("seg 5 min:  ", np.mean(np.min(seg5)))
print(" ")
print("seg 6 mean: ", np.mean(np.array(seg6)))
print("seg 6 std:  ", np.mean(np.std(seg6)))
print("seg 6 max:  ", np.mean(np.max(seg6)))
print("seg 6 min:  ", np.mean(np.min(seg6)))
print(" ")
print("seg 7 mean: ", np.mean(np.array(seg7)))
print("seg 7 std:  ", np.mean(np.std(seg7)))
print("seg 7 max:  ", np.mean(np.max(seg7)))
print("seg 7 min:  ", np.mean(np.min(seg7)))
print(" ")
print("seg 8 mean: ", np.mean(np.array(seg8)))
print("seg 8 std:  ", np.mean(np.std(seg8)))
print("seg 8 max:  ", np.mean(np.max(seg8)))
print("seg 8 min:  ", np.mean(np.min(seg8)))

print("seg 2 < 0.95: ", np.sum(np.array(seg2) < 0.95))
print("seg 2 < 0.90: ", np.sum(np.array(seg2) < 0.90))
print(" ")
print("seg 3 < 0.95: ", np.sum(np.array(seg3) < 0.95))
print("seg 3 < 0.90: ", np.sum(np.array(seg3) < 0.90))
print(" ")
print("seg 4 < 0.95: ", np.sum(np.array(seg4) < 0.95))
print("seg 4 < 0.90: ", np.sum(np.array(seg4) < 0.90))
print(" ")
print("seg 5 < 0.95: ", np.sum(np.array(seg5) < 0.95))
print("seg 5 < 0.90: ", np.sum(np.array(seg5) < 0.90))
print(" ")
print("seg 6 < 0.95: ", np.sum(np.array(seg6) < 0.95))
print("seg 6 < 0.90: ", np.sum(np.array(seg6) < 0.90))
print(" ")
print("seg 7 < 0.95: ", np.sum(np.array(seg7) < 0.95))
print("seg 7 < 0.90: ", np.sum(np.array(seg7) < 0.90))
print(" ")
print("seg 8 < 0.95: ", np.sum(np.array(seg8) < 0.95))
print("seg 8 < 0.90: ", np.sum(np.array(seg8) < 0.90))
print(" ")
print("Totali :", len(seg2))

liver_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dataset_folder,files_name[5])))
f = find_lines(liver_seg)
f = f['2,4']
reg = LinearRegression()
reg.fit(f[:,[0,2]],f[:,1])
couples = np.array([[i, j] for i in range(liver_seg.shape[0]) for j in range(liver_seg.shape[2])])
yp = np.array([couples[:,0]*reg.coef_[0] + couples[:,1]*reg.coef_[1] + reg.intercept_])
res = np.concatenate((couples, np.transpose(yp)), 1)
res = res.astype(np.int32)
res = np.array([[i, j, k] for i, j, k in res if 0 <= k < liver_seg.shape[1]])
plane_volume = np.zeros(liver_seg.shape)
plane_volume[res[:, 0], res[:, 2], res[:, 1]] = 1
for i in range(liver_seg.shape[0]):
    plt.imshow(plane_volume[i]);
    plt.title(i)
    plt.show()

couples = np.array([[i, j] for i in range(liver_seg.shape[0]) for j in range(liver_seg.shape[2])])
points_plane = {}
plane_volumes = []
for key, plane in zip(f["planes"], f["planes"].values()):
    plane_volume = np.zeros(liver_seg.shape)
    reg = plane["plane"]
    yp = np.array([couples[:, 0] * reg[0] + couples[:, 1] * reg[1] + reg[2]])
    res = np.concatenate((couples, np.transpose(yp)), 1)
    res = res.astype(np.int32)
    res = np.array([[i, j, k] for i, j, k in res if 0 <= k < liver_seg.shape[1]])
    plane_volume[res[:, 0], res[:, 2], res[:, 1]] = 1
    plane_volumes.append(plane_volume)

for plane in plane_volumes:
    plt.imshow(plane[0])
    plt.show()

g = extract_contours((8,7), liver_seg[33], 2)
plt.imshow(g)
plt.show()
plt.imshow(liver_seg[33])
plt.show()

for slice in range(liver_seg.shape[0]):
    g = extract_contours((2,4), liver_seg[slice], 5)
    plt.imshow(g); plt.title(slice); plt.show()

