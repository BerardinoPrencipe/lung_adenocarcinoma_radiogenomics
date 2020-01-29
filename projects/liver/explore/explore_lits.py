import nibabel as nib
import sys
import os


current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))


def main():
    local_path_lits = 'datasets/LiTS/train'

    lits_dataset_path = os.path.join(current_path_abs, local_path_lits)
    print('Loading LiTS training set from {}'.format(lits_dataset_path))

    images_and_masks = os.listdir(local_path_lits)
    mask_paths = [mask for mask in images_and_masks if mask[:12] == 'segmentation']


    pos_voxels_tot = 0
    neg_voxels_tot = 0

    for idx, mask_path in enumerate(mask_paths):
        print('Index: {} on {}'.format(idx, len(mask_paths)-1))
        full_mask_path = os.path.join(lits_dataset_path, mask_path)
        mask = nib.load(full_mask_path)
        mask = mask.get_data()

        pos_voxels = (mask>0).sum()
        neg_voxels = (mask==0).sum()

        pos_voxels_tot += pos_voxels
        neg_voxels_tot += neg_voxels

        print('Pos voxels = {}'.format(pos_voxels))
        print('Neg voxels = {}'.format(neg_voxels))

    print('\n\n')
    print('Positive voxels tot = {}'.format(pos_voxels_tot))
    print('Negative voxels tot = {}'.format(neg_voxels_tot))


# python3 projects/liver/explore/explore_lits.py
if __name__ == "__main__":
    main()

