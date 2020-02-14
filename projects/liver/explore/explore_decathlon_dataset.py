import os
import SimpleITK as sitk

from utils import get_num_from_path

dataset_path_base = 'E:/Datasets/LiverDecathlon'
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


# for image_filt_path in images_filtered_by_segments:
image_filt_path = images_filtered_by_segments[0]
image_filt_full_path = os.path.join(images_folder, image_filt_path)
image_filt_sitk = sitk.ReadImage(image_filt_full_path)
image_filt_np = sitk.GetArrayFromImage(image_filt_sitk)
print('Shape = {}'.format(image_filt_np.shape))