import os
import numpy as np
import SimpleITK as sitk
import torch

from utils import get_num_from_path

dataset_path_base = 'datasets/LiverDecathlon'
dataset_path = os.path.join(dataset_path_base, 'nii')
images_folder = os.path.join(dataset_path, 'images')
labels_segments_folder = os.path.join(dataset_path, 'labels_segments')
labels_liver_folder = os.path.join(dataset_path, 'labels_liver')
labels_vessels_tumors = os.path.join(dataset_path, 'labels_vessels_tumors')

images_list_paths = os.listdir(images_folder)
labels_liver_list_paths = os.listdir(labels_liver_folder)
labels_segments_list_paths = os.listdir(labels_segments_folder)
labels_vessels_tumors_list_paths = os.listdir(labels_vessels_tumors)

###################
### IMAGES IDXS ###
###################
images_nums = [get_num_from_path(image_path) for image_path in images_list_paths]

######################################
### VESSELS AND TUMORS LABELS IDXS ###
######################################
labels_vessels_tumors_nums = [get_num_from_path(label_vessel_tumor_path) for
                              label_vessel_tumor_path in labels_vessels_tumors_list_paths]

images_minus_vessels = list(set(images_nums)-set(labels_vessels_tumors_nums))
vessels_minus_images = list(set(labels_vessels_tumors_nums)-set(images_nums))
if len(images_minus_vessels) == 0 and len(vessels_minus_images) == 0:
    print('Perfect match between Images and Vessels Labels!')
else:
    print('Images - Vessels = {}'.format(images_minus_vessels))
    print('Vessels - Images = {}'.format(vessels_minus_images))


#########################
### LIVER LABELS IDXS ###
#########################
labels_liver_nums = [get_num_from_path(label_liver_path)
                     for label_liver_path in labels_liver_list_paths]

images_minus_livers = list(set(images_nums)-set(labels_liver_nums))
livers_minus_images = list(set(labels_liver_nums)-set(images_nums))
if len(images_minus_livers) == 0 and len(livers_minus_images) == 0:
    print('Perfect match between Images and Liver Labels!')
else:
    print('Images - Livers = {}'.format(images_minus_livers))
    print('Livers - Images = {}'.format(livers_minus_images))

############################
### SEGMENTS LABELS IDXS ###
############################
labels_segments_nums = [get_num_from_path(label_segment_path)
                        for label_segment_path in labels_segments_list_paths]

images_minus_segments = list(set(images_nums)-set(labels_segments_nums))
segments_minus_images = list(set(labels_segments_nums)-set(images_nums))
if len(images_minus_segments) == 0 and len(segments_minus_images) == 0:
    print('Perfect match between Images and Segments Labels!')
else:
    print('Images - Segments = {}'.format(images_minus_segments))
    print('Segments - Images = {}'.format(segments_minus_images))


print('Len Images   Nums    = {}'.format(len(images_nums)))
print('Len Livers   Nums    = {}'.format(len(labels_liver_nums)))
print('Len Vessels  Nums    = {}'.format(len(labels_vessels_tumors_nums)))
print('Len Segments Nums    = {}'.format(len(labels_segments_nums)))


images_filtered_by_segments = [image_path for image_path in images_list_paths
                               if get_num_from_path(image_path) in labels_segments_nums]
print('Len Images(Segments) = {}'.format(len(images_filtered_by_segments)))

images_filtered_by_vessels  = [image_path for image_path in images_list_paths
                               if get_num_from_path(image_path) in labels_vessels_tumors_nums]
print('Len Images(Vessels)  = {}'.format(len(images_filtered_by_vessels)))

print('Len (Segments - VesselsAndTumors) = {}'.format(len(set(labels_segments_nums) - set(labels_vessels_tumors_nums))))

################################
### GET WEIGHTS FOR SEGMENTS ###
################################
num_classes = 9
cnt_cls = np.zeros(num_classes, dtype=np.uint64)

# image_filt_path = images_filtered_by_segments[0]
for idx, image_filt_path in enumerate(images_filtered_by_segments):
    image_filt_full_path = os.path.join(labels_segments_folder, image_filt_path)
    image_filt_sitk = sitk.ReadImage(image_filt_full_path)
    image_filt_np = sitk.GetArrayFromImage(image_filt_sitk)
    print('Index {} on {}'.format(idx, len(images_filtered_by_segments)-1))
    print('Path  = {}'.format(image_filt_full_path))
    print('Shape = {}'.format(image_filt_np.shape))
    for cls in range(num_classes):
        cnt_cls[cls] += (image_filt_np == cls).sum()

sum_cls = sum(cnt_cls)
# inverse_sum_cnt_cls = [1/cnt_cls_cls for cnt_cls_cls in cnt_cls]
ratio_cnt_cls = [cnt_cls_cls/sum_cls for cnt_cls_cls in cnt_cls]
inverse_ratio_cnt_cls = [1/ratio_cnt_cl for ratio_cnt_cl in ratio_cnt_cls]
log_i_r_cnt_cls = [np.log(i_r_cnt_cl) for i_r_cnt_cl in inverse_ratio_cnt_cls]
for cls in range(num_classes):
    print("Class [{:02d}] - Voxels: {:12d} - Ratio: {:.4f} - Inverse: {:9.4f} - Log: {:9.4f}".
          format(cls, cnt_cls[cls], ratio_cnt_cls[cls], inverse_ratio_cnt_cls[cls], log_i_r_cnt_cls[cls]))

weights_balancing_path = 'logs/segments/weights.pt'
torch_weights = torch.from_numpy(np.array(log_i_r_cnt_cls))
torch.save(torch_weights, weights_balancing_path)
torch_weights_load = torch.load(weights_balancing_path)

######################################
### GET WEIGHTS FOR VESSELS_TUMORS ###
######################################
num_classes = 3
cnt_cls = np.zeros(num_classes, dtype=np.uint64)

for idx, image_filt_path in enumerate(images_filtered_by_vessels):
    image_filt_full_path = os.path.join(labels_vessels_tumors, image_filt_path)
    image_filt_sitk = sitk.ReadImage(image_filt_full_path)
    image_filt_np = sitk.GetArrayFromImage(image_filt_sitk)
    print('Index {} on {}'.format(idx, len(images_filtered_by_vessels)-1))
    print('Path  = {}'.format(image_filt_full_path))
    print('Shape = {}'.format(image_filt_np.shape))
    for cls in range(num_classes):
        cnt_cls[cls] += (image_filt_np == cls).sum()

sum_cls = sum(cnt_cls)
ratio_cnt_cls = [cnt_cls_cls/sum_cls for cnt_cls_cls in cnt_cls]
inverse_ratio_cnt_cls = [1/ratio_cnt_cl for ratio_cnt_cl in ratio_cnt_cls]
log_i_r_cnt_cls = [np.log(i_r_cnt_cl) for i_r_cnt_cl in inverse_ratio_cnt_cls]
for cls in range(num_classes):
    print("Class [{:02d}] - Voxels: {:12d} - Ratio: {:.4f} - Inverse: {:9.4f} - Log: {:9.4f}".
          format(cls, cnt_cls[cls], ratio_cnt_cls[cls], inverse_ratio_cnt_cls[cls], log_i_r_cnt_cls[cls]))

weights_balancing_path = 'logs/vessels_tumors/weights.pt'
torch_weights = torch.from_numpy(np.array(log_i_r_cnt_cls))
torch.save(torch_weights, weights_balancing_path)
torch_weights_load = torch.load(weights_balancing_path)
