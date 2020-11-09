import json
import os
import pickle as pk
import numpy as np
import SimpleITK as sitk
from projects.liver.geometric.points_segments import find_lines_one, extract_segments

liver_folder   = "D:\\Universita\\Laurea Magistrale - Computer Science Engeneering\\Tesi\\LiverSegmentation\\datasets\\LiverDecathlon\\nii\\labels_liver\\"
dataset_folder = "D:\\Universita\\Laurea Magistrale - Computer Science Engeneering\\Tesi\\LiverSegmentation\\datasets/LiverDecathlon/nii/labels_segments"
result_folder  = "D:\\Universita\\Laurea Magistrale - Computer Science Engeneering\\Tesi\\LiverSegmentation\\datasets\\LiverDecathlon\\nii\\rebulit_label\\"
files_name = os.listdir(dataset_folder)

# Ground truth without portal vein indication
files_name.remove('hepaticvessel_005.nii.gz')
files_name.remove('hepaticvessel_063.nii.gz')
files_name.remove('hepaticvessel_133.nii.gz')
files_name.remove('hepaticvessel_159.nii.gz')
files_name.remove('hepaticvessel_237.nii.gz')
files_name.remove('hepaticvessel_347.nii.gz')


results = {}
for name in files_name:
    if name in ["hepaticvessel_005.nii.gz", "hepaticvessel_063.nii.gz", "hepaticvessel_133.nii.gz",
                "hepaticvessel_159.nii.gz", "hepaticvessel_237.nii.gz", "hepaticvessel_347.nii.gz"]:
        results[name] = None
        continue
    liver_seg_sitk = sitk.ReadImage(os.path.join(dataset_folder,name))
    liver_seg = sitk.GetArrayFromImage(liver_seg_sitk)
    liver_seg = liver_seg.astype(np.uint8)
    result = find_lines_one(liver_seg)
    results[name] = result
    print(name)

results_json = {}
for result,name in zip(results.values(), files_name):
    if name in ["hepaticvessel_005.nii.gz", "hepaticvessel_063.nii.gz", "hepaticvessel_133.nii.gz",
                "hepaticvessel_159.nii.gz", "hepaticvessel_237.nii.gz", "hepaticvessel_347.nii.gz"]:
        results_json[name] = None
        continue
    r = {}
    lpv = int(result["left_pv"])
    rpv = int(result["right_pv"])
    lines = result["lines"]
    ls = []
    for line in lines:
        l = {}
        t = line["line"]
        e = []
        for el in t:
            e.append(float(el))
        e = tuple(e)
        r2 = float(line["r2"])
        l["line"] = e
        l["r2"] = r2
        ls.append(l)
    r["left_pv"] = lpv
    r["right_pv"] = rpv
    r["lines"] = ls
    results_json[name] = r



json.dump([{"id1": files_name},{"id2":["left_pv", "right_pv", "lines"]},{"id3":["line","r2"]}, results_json], open("results.json", "w"))
json.dump(results_json, open("results2.json", "w"))

pk.dump(results, open("results.pkl", "wb"))

# results = pk.load(open("results.pkl", "rb"))

for name,result in zip(files_name,results):
    lines = [r["line"] for r in result["lines"]]
    liver_sitk = sitk.ReadImage(os.path.join(liver_folder, name))
    liver = sitk.GetArrayFromImage(liver_sitk)
    liver_rebuilt = extract_segments(lines, liver, lpv=result["left_pv"], rpv=result["right_pv"])
    liver_rebuilt_sitk = sitk.GetImageFromArray(liver_rebuilt)
    liver_rebuilt_sitk.SetSpacing(liver_sitk.GetSpacing())
    liver_rebuilt_sitk.SetOrigin(liver_sitk.GetOrigin())
    liver_rebuilt_sitk.SetDirection(liver_sitk.GetDirection())
    writer = sitk.ImageFileWriter()
    writer.SetFileName(os.path.join(result_folder, name))
    writer.Execute(liver_rebuilt_sitk)
    print(os.path.join(name))

q_list = []
m_list = []
for result in results:
    for line in result["lines"]:
        q_list.append(line["line"][2])
        m_list.append(line["line"][0])
print("max m:",max(m_list))
print("min m:",min(m_list))
print("max q:",max(q_list))
print("min q:",min(q_list))

files = []
for name in files_name:
    liver_seg_sitk = sitk.ReadImage(os.path.join(dataset_folder,name))
    liver_seg = sitk.GetArrayFromImage(liver_seg_sitk)
    liver_rebuilt_sitk = sitk.ReadImage(os.path.join(result_folder,name))
    liver_rebuilt = sitk.GetArrayFromImage(liver_rebuilt_sitk)
    liver_rebuilt[liver_seg == 1] = 0
    values = [2,3,4,5,6,7,8]
    DSC = {}
    for value in values:
        l = liver_seg.copy()
        lr = liver_rebuilt.copy()
        l[l != value] = 0
        lr[lr != value] = 0
        a = (np.sum(np.logical_and(l, lr)))
        b = (np.sum(l > 0) + np.sum(lr > 0))
        DSC[value] = 2*a/b
    files.append(DSC)
    print(name)

r1 = pk.load(open("files.pkl", "rb"))
idx = [7,29,30,31,38,45,50,56,57,61,66,71,82,84,89,92,104,110,112,118,119,123,124,140,141,173,180,182,184]

for i in reversed(idx):
    del r[i]

# r.remove(7)
# r.remove(29)
# r.remove(30)
# r.remove(31)
# r.remove(38)
# r.remove(45)
# r.remove(50)
# r.remove(56)
# r.remove(57)
# r.remove(61)
# r.remove(66)
# r.remove(71)
# r.remove(82)
# r.remove(84)
# r.remove(89)
# r.remove(92)
# r.remove(104)
# r.remove(110)
# r.remove(112)
# r.remove(118)
# r.remove(119)
# r.remove(123)
# r.remove(124)
# r.remove(140)
# r.remove(141)
# r.remove(173)
# r.remove(180)
# r.remove(182)
# r.remove(184)


seg2 = [a[2] for a in r]
seg3 = [a[3] for a in r]
seg4 = [a[4] for a in r]
seg5 = [a[5] for a in r]
seg6 = [a[6] for a in r]
seg7 = [a[7] for a in r]
seg8 = [a[8] for a in r]

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

print("seg 3 < 0.7: ", np.sum(np.array(seg3) < 0.7))
print("seg 5 < 0.7: ", np.sum(np.array(seg5) < 0.7))
print("seg 6 < 0.7: ", np.sum(np.array(seg6) < 0.7))


image_folder = "./datasets/LiverDecathlon/nii/images/"

dimensions = []
files_name = os.listdir(image_folder)
for name in os.listdir(image_folder):
    img = sitk.ReadImage(os.path.join(image_folder,name))
    dimensions.append(img.GetSpacing())
    print(name)
dimensions1 = np.array(dimensions)


