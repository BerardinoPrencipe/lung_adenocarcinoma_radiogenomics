import SimpleITK as sitk
import numpy as np


def get_largest_cc(image):
    # Get connected components
    ccif = sitk.ConnectedComponentImageFilter()
    sitk_output = sitk.GetImageFromArray(image)
    conn_comps = ccif.Execute(sitk_output)
    conn_comps_np = sitk.GetArrayFromImage(conn_comps)

    unique_values = np.unique(conn_comps_np)
    n_uniques = len(unique_values)
    counter_uniques = np.zeros(n_uniques)
    for i in range(1, max(unique_values) + 1):
        counter_uniques[i] = (conn_comps_np == i).sum()
    biggest_region_value = np.argmax(counter_uniques)

    # Get largest connected component
    largest_conn_comp = np.zeros(conn_comps_np.shape, dtype=np.uint8)
    largest_conn_comp[conn_comps_np == biggest_region_value] = 1

    return largest_conn_comp

def post_process_liver(output, vector_radius=(25, 25, 25), kernel=sitk.sitkBall):
    '''
    # Get connected components
    ccif = sitk.ConnectedComponentImageFilter()
    sitk_output = sitk.GetImageFromArray(output)
    conn_comps = ccif.Execute(sitk_output)
    conn_comps_np = sitk.GetArrayFromImage(conn_comps)

    unique_values = np.unique(conn_comps_np)
    n_uniques = len(unique_values)
    counter_uniques = np.zeros(n_uniques)
    for i in range(1, max(unique_values) + 1):
        counter_uniques[i] = (conn_comps_np == i).sum()
    biggest_region_value = np.argmax(counter_uniques)

    # Get largest connected component
    largest_conn_comp = np.zeros(conn_comps_np.shape, dtype=np.uint8)
    largest_conn_comp[conn_comps_np == biggest_region_value] = 1
    '''

    largest_conn_comp = get_largest_cc(output)

    # Morphological Closing
    largest_conn_comp_uint8 = largest_conn_comp.astype(np.uint8)
    largest_conn_comp_sitk = sitk.GetImageFromArray(largest_conn_comp_uint8)
    largest_conn_comp_closed_sitk = sitk.BinaryMorphologicalClosing(largest_conn_comp_sitk, vector_radius, kernel)
    largest_conn_comp_closed_np = sitk.GetArrayFromImage(largest_conn_comp_closed_sitk)

    # Output
    print("Liver Voxels in Original Output    : ", output.sum())
    print("Liver Voxels after ConnectCompLabel: ", largest_conn_comp.sum())
    print("Liver Voxels after Closing         : ", largest_conn_comp_closed_np.sum())

    return largest_conn_comp_closed_np

NUM_SEGMENTS = 8
segments = [idx+1 for idx in range(NUM_SEGMENTS)]

# M[I,J] = 1 if the segment i and j can be in the same slice, otherwise 0
segments_consistency_matrix = np.resize(np.array([1,1,1,1,1,1,1,1,
                                                  1,1,0,1,0,0,1,1,
                                                  1,0,1,1,1,1,1,1,
                                                  1,1,1,1,1,1,1,1,
                                                  1,0,1,1,1,1,0,0,
                                                  1,0,1,1,1,1,0,0,
                                                  1,1,1,1,0,0,1,1,
                                                  1,1,1,1,0,0,1,1]),(NUM_SEGMENTS,NUM_SEGMENTS))

segments_other    = [1,4]
segments_above_pv = [2,7,8]
segments_below_pv = [3,5,6]

segments_above_left_pv  = [2]
segments_above_right_pv = [7,8]
segments_below_left_pv  = [3]
segments_below_right_pv = [5,6]


def consistency_check(slice):
    inconsistent_pairs = []
    slice = np.array(slice,dtype=np.uint8).flatten()
    labels = np.unique(slice)
    labels = list( set(labels) - {0} )
    labels.sort()
    assert max(labels) <= NUM_SEGMENTS and min(labels) > 0, 'Labels out of range {}'.format(labels)
    for idx_i, i in enumerate(labels):
        for j in labels[idx_i+1:]:
            print(i, j)
            if segments_consistency_matrix[i-1,j-1] == 0:
                inconsistent_pairs.append((i,j))

    is_consistent = True if len(inconsistent_pairs) == 0 else False
    return is_consistent, inconsistent_pairs

def check_above_below_pv(slice):
    slice = np.array(slice, dtype=np.uint8).flatten()

    # COUNT VOXELS ABOVE PV
    cnt_above = 0
    for idx_above_pv in segments_above_pv:
        cnt_above += (slice == idx_above_pv).sum()

    # COUNT VOXELS BELOW PV
    cnt_below = 0
    for idx_below_pv in segments_below_pv:
        cnt_below += (slice == idx_below_pv).sum()

    is_above_pv = cnt_above > cnt_below
    return is_above_pv, cnt_above, cnt_below

def check_above_below_left_pv(slice):
    slice = np.array(slice, dtype=np.uint8).flatten()

    # COUNT VOXELS ABOVE LEFT PV
    cnt_above = 0
    for idx_above_pv in segments_above_left_pv:
        cnt_above += (slice == idx_above_pv).sum()

    # COUNT VOXELS BELOW LEFT PV
    cnt_below = 0
    for idx_below_pv in segments_below_left_pv:
        cnt_below += (slice == idx_below_pv).sum()

    is_above_pv = cnt_above > cnt_below
    return is_above_pv, cnt_above, cnt_below

def check_above_below_right_pv(slice):
    slice = np.array(slice, dtype=np.uint8).flatten()

    # COUNT VOXELS ABOVE RIGHT PV
    cnt_above = 0
    for idx_above_pv in segments_above_right_pv:
        cnt_above += (slice == idx_above_pv).sum()

    # COUNT VOXELS BELOW RIGHT PV
    cnt_below = 0
    for idx_below_pv in segments_below_right_pv:
        cnt_below += (slice == idx_below_pv).sum()

    is_above_pv = cnt_above > cnt_below
    return is_above_pv, cnt_above, cnt_below

def correct_slice_left_pv(slice):
    is_above_left_pv, _, _ = check_above_below_left_pv(slice)
    if is_above_left_pv:
        slice[slice == 3] = 2
    else:
        slice[slice == 2] = 3
    return slice

def correct_slice_right_pv(slice):
    is_above_right_pv, _, _ = check_above_below_right_pv(slice)
    if is_above_right_pv:
        slice[slice == 6] = 7
        slice[slice == 5] = 8
    else:
        slice[slice == 7] = 6
        slice[slice == 8] = 5
    return slice

def correct_slice_right_left_pv(slice):
    slice_corr_right = correct_slice_right_pv(slice)
    slice_corr = correct_slice_left_pv(slice_corr_right)
    return slice_corr

def correct_slice(slice):
    is_above_pv, _, _ = check_above_below_pv(slice)
    if is_above_pv:
        slice[slice == 3] = 2
        slice[slice == 6] = 7
        slice[slice == 5] = 8
    else:
        slice[slice == 2] = 3
        slice[slice == 7] = 6
        slice[slice == 8] = 5
    return slice


def correct_volume(volume):
    corr_volume = np.zeros(volume.shape, dtype=np.uint8)
    for idx, slice in enumerate(volume):
        slice_corr = correct_slice(slice)
        corr_volume[idx] = slice_corr
    return corr_volume


def correct_volume_right_left(volume):
    corr_volume = np.zeros(volume.shape, dtype=np.uint8)
    for idx, slice in enumerate(volume):
        slice_corr = correct_slice_right_left_pv(slice)
        corr_volume[idx] = slice_corr
    return corr_volume

def correct_volume_slice_split(volume):
    corr_volume = volume.copy()
    index_split_left, index_split_right = -1, -1
    for idx, slice in enumerate(volume):
        is_above_right_pv, _, _ = check_above_below_right_pv(slice)
        is_above_left_pv , _, _ = check_above_below_left_pv(slice)
        if is_above_left_pv and index_split_left == -1:
            index_split_left  = idx
        if is_above_right_pv and index_split_left == -1:
            index_split_right = idx
        if index_split_left > 0 and index_split_right > 0:
            break

    print('Volume shape      = {}'.format(volume.shape))
    print('Index Split Left  = {}'.format(index_split_left))
    print('Index Split Right = {}'.format(index_split_right))

    # BELOW LEFT
    corr_volume[0:index_split_left, :, :][volume[0:index_split_left, :, :] == 2] = 3

    # ABOVE LEFT
    corr_volume[index_split_left:, :, :][volume[index_split_left:, :, :] == 3] = 2

    # BELOW RIGHT
    corr_volume[0:index_split_right, :, :][volume[0:index_split_right, :, :] == 7] = 6
    corr_volume[0:index_split_right, :, :][volume[0:index_split_right, :, :] == 8] = 5

    # ABOVE RIGHT
    corr_volume[index_split_right:, :, :][volume[index_split_right:, :, :] == 6] = 7
    corr_volume[index_split_right:, :, :][volume[index_split_right:, :, :] == 5] = 8

    return corr_volume

def get_complement(idx):
    if idx == 2:
        return 3
    elif idx == 3:
        return 2
    elif idx == 6:
        return 7
    elif idx == 7:
        return 6
    elif idx == 8:
        return 5
    elif idx == 5:
        return 8
    else:
        return idx

def erase_non_max_cc_segments(image):
    image_after_erase = np.zeros(image.shape, dtype=np.uint8)
    for idx_segment in segments:
        image_idx = (image == idx_segment).astype(np.uint8)
        image_idx_max_cc = get_largest_cc(image_idx)
        image_after_erase[image_idx_max_cc!=0] = idx_segment
    return image_after_erase


