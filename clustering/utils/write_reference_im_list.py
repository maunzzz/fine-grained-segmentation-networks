import os
import sys
import h5py
import numpy as np

import context
import datasets.dataset_configs as data_configs


def write_reference_im_list(corr_set):
    if corr_set == 'cmu':
        corr_set_config = data_configs.CmuConfig()
    elif corr_set == 'rc':
        corr_set_config = data_configs.RobotcarConfig()

    # FILES FROM LIST
    f_name_list = []
    with open(corr_set_config.correspondence_train_list_file) as f:
        for line in f:
            f_name_list.append(line.strip())

    # other
    ref_file_names = set()  # use set to automatically handle duplicates

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
        ref_file_names.add(im1name)
        it += 1

        if (it % 100) == 0:
            print('%d/%d' % (it, len(f_name_list)))

    print('Writing file %s ' % corr_set_config.reference_image_list)
    with open(corr_set_config.reference_image_list, 'w') as f:
        for fname in ref_file_names:
            f.write('%s\n' % fname)
