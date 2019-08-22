import context
import faulthandler
from utils.segmentor import Segmentor
from utils.misc import check_mkdir, get_global_opts
from models import model_configs
from datasets import correspondences, merged
import utils.corr_transforms as corr_transforms
import utils.transforms as extended_transforms
import utils.joint_transforms as joint_transforms
import datasets.dataset_configs as data_configs
import datetime
import os
import re
import sys
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F
from shutil import copyfile
from torch import optim
import torchvision.transforms as standard_transforms


def find_non_stationary_clusters(args):
    if args['use_gpu']:
        print("Using CUDA" if torch.cuda.is_available() else "Using CPU")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    network_folder_name = args['folder'].split('/')[-1]
    tmp = re.search(r"cn(\d+)-", network_folder_name)
    n_clusters = int(tmp.group(1))

    save_folder = os.path.join(args['dest_root'], network_folder_name)
    if os.path.exists(os.path.join(save_folder, 'cluster_histogram_for_corr.npy')):
        print('{} already exists. skipping'.format(
            os.path.join(save_folder, 'cluster_histogram_for_corr.npy')))
        return

    check_mkdir(save_folder)

    with open(os.path.join(args['folder'], 'bestval.txt')) as f:
        best_val_dict_str = f.read()
        bestval = eval(best_val_dict_str.rstrip())

    # Network and weight loading
    model_config = model_configs.PspnetCityscapesConfig()
    net = model_config.init_network(
        n_classes=n_clusters, for_clustering=False, output_features=False, use_original_base=args['use_original_base']).to(device)
    net.load_state_dict(torch.load(os.path.join(
        args['folder'], bestval['snapshot'] + '.pth')))  # load weights
    net.eval()

    # copy network file to save location
    copyfile(os.path.join(args['folder'], bestval['snapshot'] + '.pth'), os.path.join(save_folder, 'weights.pth'))

    if args['only_copy_weights']:
        print('Only copying weights')
        return

    # Data loading setup
    if args['corr_set'] == 'rc':
        corr_set_config = data_configs.RobotcarConfig()
    elif args['corr_set'] == 'cmu':
        corr_set_config = data_configs.CmuConfig()
    elif args['corr_set'] == 'both':
        corr_set_config1 = data_configs.CmuConfig()
        corr_set_config2 = data_configs.RobotcarConfig()

    sliding_crop_im = joint_transforms.SlidingCropImageOnly(
        713, args['stride_rate'])

    input_transform = model_config.input_transform
    pre_validation_transform = model_config.pre_validation_transform

    if args['corr_set'] == 'both':
        corr_set_val1 = correspondences.Correspondences(corr_set_config1.correspondence_path, corr_set_config1.correspondence_im_path, input_size=(713, 713),
                                                        input_transform=None, joint_transform=None, listfile=corr_set_config1.correspondence_val_list_file)
        corr_set_val2 = correspondences.Correspondences(corr_set_config2.correspondence_path, corr_set_config2.correspondence_im_path, input_size=(713, 713),
                                                        input_transform=None, joint_transform=None, listfile=corr_set_config2.correspondence_val_list_file)
        corr_set_val = merged.Merged([corr_set_val1, corr_set_val2])
    else:
        corr_set_val = correspondences.Correspondences(corr_set_config.correspondence_path, corr_set_config.correspondence_im_path, input_size=(713, 713),
                                                       input_transform=None, joint_transform=None, listfile=corr_set_config.correspondence_val_list_file)

    # Segmentor
    segmentor = Segmentor(net, n_clusters, n_slices_per_pass=4)

    # save args
    open(os.path.join(save_folder, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')

    cluster_histogram_for_correspondences = np.zeros((n_clusters,), dtype=np.int64)
    cluster_histogram_non_correspondences = np.zeros((n_clusters,), dtype=np.int64)

    for i in range(0, len(corr_set_val), args['step']):
        img1, img2, pts1, pts2, _ = corr_set_val[i]
        seg1 = segmentor.run_and_save(img1, None, pre_sliding_crop_transform=pre_validation_transform,
                                      input_transform=input_transform, sliding_crop=sliding_crop_im, use_gpu=args['use_gpu'])

        seg1 = np.array(seg1)
        corr_loc_mask = np.zeros(seg1.shape, dtype=np.bool)

        valid_inds = (pts1[0, :] >= 0) & (pts1[0, :] < seg1.shape[1]) & (
            pts1[1, :] >= 0) & (pts1[1, :] < seg1.shape[0])

        pts1 = pts1[:, valid_inds]
        for j in range(pts1.shape[1]):
            pt = pts1[:, j]
            corr_loc_mask[pt[1], pt[0]] = True

        cluster_ids_corr = seg1[corr_loc_mask]
        hist_tmp, _ = np.histogram(cluster_ids_corr, np.arange(n_clusters+1))
        cluster_histogram_for_correspondences += hist_tmp

        cluster_ids_no_corr = seg1[~corr_loc_mask]
        hist_tmp, _ = np.histogram(
            cluster_ids_no_corr, np.arange(n_clusters+1))
        cluster_histogram_non_correspondences += hist_tmp

        if ((i + 1) % 100) < args['step']:
            print('{}/{}'.format(i + 1, len(corr_set_val)))

    np.save(os.path.join(save_folder, 'cluster_histogram_for_corr.npy'), cluster_histogram_for_correspondences)
    np.save(os.path.join(save_folder, 'cluster_histogram_non_corr.npy'), cluster_histogram_non_correspondences)
    frac = cluster_histogram_for_correspondences / \
        (cluster_histogram_for_correspondences +
         cluster_histogram_non_correspondences)
    stationary_inds = np.argwhere(frac > 0.01)
    np.save(os.path.join(save_folder, 'stationary_inds.npy'), stationary_inds)
    print('{} stationary clusters out of {}'.format(len(stationary_inds), len(cluster_histogram_for_correspondences)))


if __name__ == '__main__':
    args = {
        # set to '' to start training from beginning and 'latest' to use last checkpoint
        'snapshot': 'latest',
        'folder': 'home/user/path/to/training/result/folder',
        'use_original_base': False,
        'only_copy_weights': False,
        'dest_root': 'home/user/networks-for-localization',
        'stride_rate': 2 / 3.,
        'use_gpu': True,
        # dataset settings
        'corr_set': 'cmu',  # 'cmu' or 'rc
        'step': 5
    }

    find_non_stationary_clusters(args)
