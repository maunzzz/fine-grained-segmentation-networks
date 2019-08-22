import sys
import os
import numpy as np
from math import ceil

import context
import datasets.dataset_configs as data_configs


def write_lists_for_corr(corr_set):
    if corr_set == 'cmu':
        corr_set_config = data_configs.CmuConfig()
    elif corr_set == 'rc':
        corr_set_config = data_configs.RobotcarConfig()

    n_samples_to_include = 1000000
    fraction_training = 0.7

    np.random.seed(0)
    f_name_list = [fn for fn in os.listdir(corr_set_config.correspondence_path) if fn.endswith('mat')]

    n_samples = min(n_samples_to_include, len(f_name_list))

    n_training = ceil(fraction_training*n_samples)
    n_validation = n_samples - n_training

    training_ids = np.random.choice(len(f_name_list), n_training)
    ids_left_for_validation = set(range(len(f_name_list))) - set(training_ids)
    validation_ids = np.random.choice(list(ids_left_for_validation), n_validation)

    f_train = open(corr_set_config.correspondence_train_list_file, 'w')
    f_val = open(corr_set_config.correspondence_val_list_file, 'w')
    for i in training_ids:
        f_train.write(f_name_list[i] + '\n')
    for i in validation_ids:
        f_val.write(f_name_list[i] + '\n')
    f_train.close()
    f_val.close()
