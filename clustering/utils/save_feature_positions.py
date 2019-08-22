import context
import datasets.dataset_configs as data_configs
from utils.misc import check_mkdir, im_to_ext_name
import os
import sys
import h5py
import numpy as np


def save_feature_positions(corr_set):
    if corr_set == 'cmu':
        corr_set_config = data_configs.CmuConfig()
    elif corr_set == 'rc':
        corr_set_config = data_configs.RobotcarConfig()

    save_dir = corr_set_config.reference_feature_poitions

    # FILES FROM LIST
    f_name_list = []
    with open(corr_set_config.correspondence_train_list_file) as f:
        for line in f:
            f_name_list.append(line.strip())

    # other
    ref_feat_positions = {}

    # run
    it = 0
    for f_name in f_name_list:
        mat_content = {}
        ff = h5py.File(os.path.join(
            corr_set_config.correspondence_path, f_name), 'r')
        for k, v in ff.items():
            mat_content[k] = np.array(v)

        im1name = ''.join(chr(a)
                          for a in mat_content['im_i_path'])  # convert to string
        ref_feature_positions = mat_content['pt_i']

        if im1name not in ref_feat_positions.keys():
            ref_feat_positions[im1name] = ref_feature_positions
        else:
            np.append(ref_feat_positions[im1name], ref_feature_positions, axis=0)

        it += 1

        if (it % 100) == 0:
            print('%d/%d' % (it, len(f_name_list)))

    # Remove all duplicates
    for key, pts in ref_feat_positions.items():
        ref_feat_positions[key] = np.unique(pts, axis=0)

    check_mkdir(save_dir)
    for key, pts in ref_feat_positions.items():
        h5_name = im_to_ext_name(key, 'h5')

        h5f = h5py.File(os.path.join(save_dir, h5_name), 'w')
        h5f.create_dataset('corr_pts', data=pts)
        h5f.close()
