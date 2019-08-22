import datetime
import os
import sys
import numpy as np
import h5py
import copy
from math import sqrt
import torch
import faiss
import torchvision.transforms as standard_transforms
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import context

import datasets.dataset_configs as data_configs
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
import utils.corr_transforms as corr_transforms

from datasets import correspondences, merged
from models import model_configs
from utils.misc import check_mkdir, AverageMeter, freeze_bn, get_global_opts, rename_keys_to_match, get_latest_network_name, clean_log_before_continuing, load_resnet101_weights, get_network_name_from_iteration
from utils.validator import CorrValidator
from layers.feature_loss import FeatureLoss
from layers.cluster_correspondence_loss import ClusterCorrespondenceLoss
from clustering import clustering
from clustering.cluster_tools import extract_features_for_reference, save_cluster_features_as_segmentations, assign_cluster_ids_to_correspondence_points
from clustering.clustering import preprocess_features


def init_last_layers(state_dict, n_clusters):
    device = state_dict['conv6.weight'].device
    state_dict['conv6.weight'] = torch.zeros([n_clusters, state_dict['conv6.weight'].size(
        1), state_dict['conv6.weight'].size(2), state_dict['conv6.weight'].size(3)],
        dtype=torch.float, device=device)
    state_dict['conv6.bias'] = torch.zeros([n_clusters], dtype=torch.float, device=device)
    state_dict['conv6_1.weight'] = torch.zeros([n_clusters, state_dict['conv6_1.weight'].size(1),
                                                state_dict['conv6_1.weight'].size(2),
                                                state_dict['conv6_1.weight'].size(3)],
                                               dtype=torch.float, device=device)
    state_dict['conv6_1.bias'] = torch.zeros([n_clusters], dtype=torch.float, device=device)


def reinit_last_layers(net):
    net.conv6.weight.data.normal_(0, 0.01)
    net.conv6_1.weight.data.normal_(0, 0.01)
    net.conv6.bias.data.zero_()
    net.conv6_1.bias.data.zero_()


def train_with_clustering(save_folder, tmp_seg_folder, startnet, args):
    print(save_folder.split('/')[-1])
    skip_clustering = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    check_mkdir(save_folder)
    writer = SummaryWriter(save_folder)
    check_mkdir(tmp_seg_folder)

    # Network and weight loading
    model_config = model_configs.PspnetCityscapesConfig()
    net = model_config.init_network(n_classes=args['n_clusters'], for_clustering=True,
                                    output_features=True, use_original_base=args['use_original_base']).to(device)

    if args['snapshot'] == 'latest':
        args['snapshot'] = get_latest_network_name(save_folder)
    if len(args['snapshot']) == 0:  # If start from beginning
        state_dict = torch.load(startnet)
        if 'resnet101' in startnet:
            load_resnet101_weights(net, state_dict)
        else:
            # needed since we slightly changed the structure of the network in pspnet
            state_dict = rename_keys_to_match(state_dict)
            init_last_layers(state_dict, args['n_clusters'])  # different amount of classes

            net.load_state_dict(state_dict)  # load original weights

        start_iter = 0
        args['best_record'] = {'iter': 0, 'val_loss_feat': 1e10, 'val_loss_out': 1e10, 'val_loss_cluster': 1e10}
    else:  # If continue training
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(save_folder, args['snapshot'])))  # load weights
        split_snapshot = args['snapshot'].split('_')

        start_iter = int(split_snapshot[1])
        with open(os.path.join(save_folder, 'bestval.txt')) as f:
            best_val_dict_str = f.read()
        args['best_record'] = eval(best_val_dict_str.rstrip())

        if start_iter >= args['max_iter']:
            return

        if (start_iter % args['cluster_interval']) == 0:
            skip_clustering = False
        else:
            skip_clustering = True
            last_cluster_network_snapshot_iter = (start_iter // args['cluster_interval']) * args['cluster_interval']

            # load cluster info
            cluster_info = {}
            f = h5py.File(os.path.join(save_folder, 'centroids_{}.h5'.format(last_cluster_network_snapshot_iter)), 'r')
            for k, v in f.items():
                cluster_info[k] = np.array(v)
            cluster_centroids = cluster_info['cluster_centroids']
            pca_info = [cluster_info['pca_transform_Amat'], cluster_info['pca_transform_bvec']]

            # load network that was used for last clustering
            net_for_clustering = model_config.init_network(
                n_classes=args['n_clusters'], for_clustering=True, output_features=True, use_original_base=args['use_original_base'])

            if last_cluster_network_snapshot_iter == 0:
                state_dict = torch.load(startnet, map_location=lambda storage, loc: storage)
                if 'resnet101' in startnet:
                    load_resnet101_weights(net_for_clustering, state_dict)
                else:
                    # needed since we slightly changed the structure of the network in pspnet
                    state_dict = rename_keys_to_match(state_dict)
                    init_last_layers(state_dict, args['n_clusters'])  # different amount of classes

                    net_for_clustering.load_state_dict(state_dict)  # load original weights
            else:
                cluster_network_weights = get_network_name_from_iteration(
                    save_folder, last_cluster_network_snapshot_iter)
                net_for_clustering.load_state_dict(torch.load(os.path.join(
                    save_folder, cluster_network_weights), map_location=lambda storage, loc: storage))  # load weights

    # Data loading setup
    if args['corr_set'] == 'rc':
        corr_set_config = data_configs.RobotcarConfig()
    elif args['corr_set'] == 'cmu':
        corr_set_config = data_configs.CmuConfig()
    elif args['corr_set'] == 'both':
        corr_set_config1 = data_configs.CmuConfig()
        corr_set_config2 = data_configs.RobotcarConfig()

    if args['corr_set'] == 'both':
        ref_image_lists = [corr_set_config1.reference_image_list, corr_set_config2.reference_image_list]
        corr_im_paths = [corr_set_config1.correspondence_im_path, corr_set_config2.correspondence_im_path]
        ref_featurs_pos = [corr_set_config1.reference_feature_poitions, corr_set_config2.reference_feature_poitions]
    else:
        ref_image_lists = [corr_set_config.reference_image_list]
        corr_im_paths = [corr_set_config.correspondence_im_path]
        ref_featurs_pos = [corr_set_config.reference_feature_poitions]

    input_transform = model_config.input_transform

    train_joint_transform_corr = corr_transforms.Compose([
        corr_transforms.CorrResize(1024),
        corr_transforms.CorrRandomCrop(713)
    ])

    # Correspondences for training
    if args['corr_set'] == 'both':
        corr_set_train1 = correspondences.Correspondences(corr_set_config1.correspondence_path,
                                                          corr_set_config1.correspondence_im_path,
                                                          input_size=(713, 713),
                                                          input_transform=input_transform,
                                                          joint_transform=train_joint_transform_corr,
                                                          listfile=corr_set_config1.correspondence_train_list_file)
        corr_set_train2 = correspondences.Correspondences(corr_set_config2.correspondence_path,
                                                          corr_set_config2.correspondence_im_path,
                                                          input_size=(713, 713),
                                                          input_transform=input_transform,
                                                          joint_transform=train_joint_transform_corr,
                                                          listfile=corr_set_config2.correspondence_train_list_file)

        corr_set_train = merged.Merged([corr_set_train1, corr_set_train2])
    else:
        corr_set_train = correspondences.Correspondences(corr_set_config.correspondence_path,
                                                         corr_set_config.correspondence_im_path,
                                                         input_size=(713, 713),
                                                         input_transform=input_transform,
                                                         joint_transform=train_joint_transform_corr,
                                                         listfile=corr_set_config.correspondence_train_list_file)

    corr_loader_train = DataLoader(corr_set_train, batch_size=1, num_workers=args['n_workers'], shuffle=True)

    # Correspondences for validation
    if args['corr_set'] == 'both':
        corr_set_val1 = correspondences.Correspondences(corr_set_config1.correspondence_path,
                                                        corr_set_config1.correspondence_im_path,
                                                        input_size=(713, 713),
                                                        input_transform=input_transform,
                                                        joint_transform=train_joint_transform_corr,
                                                        listfile=corr_set_config1.correspondence_val_list_file)

        corr_set_val2 = correspondences.Correspondences(corr_set_config2.correspondence_path,
                                                        corr_set_config2.correspondence_im_path,
                                                        input_size=(713, 713),
                                                        input_transform=input_transform,
                                                        joint_transform=train_joint_transform_corr,
                                                        listfile=corr_set_config2.correspondence_val_list_file)

        corr_set_val = merged.Merged([corr_set_val1, corr_set_val2])
    else:
        corr_set_val = correspondences.Correspondences(corr_set_config.correspondence_path,
                                                       corr_set_config.correspondence_im_path,
                                                       input_size=(713, 713),
                                                       input_transform=input_transform,
                                                       joint_transform=train_joint_transform_corr,
                                                       listfile=corr_set_config.correspondence_val_list_file)

    corr_loader_val = DataLoader(corr_set_val, batch_size=1, num_workers=args['n_workers'], shuffle=False)

    # Loss setup
    val_corr_loss_fct_feat = FeatureLoss(input_size=[713, 713],
                                         loss_type=args['feature_distance_measure'],
                                         feat_dist_threshold_match=0.8,
                                         feat_dist_threshold_nomatch=0.2,
                                         n_not_matching=0)

    val_corr_loss_fct_out = FeatureLoss(input_size=[713, 713],
                                        loss_type='KL',
                                        feat_dist_threshold_match=0.8,
                                        feat_dist_threshold_nomatch=0.2,
                                        n_not_matching=0)

    loss_fct = ClusterCorrespondenceLoss(input_size=[713, 713], size_average=True).to(device)
    seg_loss_fct = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')

    if args['feature_hinge_loss_weight'] > 0:
        feature_loss_fct = FeatureLoss(input_size=[713, 713],
                                       loss_type='hingeF',
                                       feat_dist_threshold_match=0.8,
                                       feat_dist_threshold_nomatch=0.2,
                                       n_not_matching=0)

    # Validator
    corr_validator = CorrValidator(corr_loader_val,
                                   val_corr_loss_fct_feat,
                                   val_corr_loss_fct_out,
                                   loss_fct,
                                   save_snapshot=True,
                                   extra_name_str='Corr')

    # Optimizer setup
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'], nesterov=True)

    # Clustering
    deepcluster = clustering.Kmeans(args['n_clusters'])
    if skip_clustering:
        deepcluster.set_index(cluster_centroids)

    if len(args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(save_folder, 'opt_' + args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    open(os.path.join(save_folder, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')

    if len(args['snapshot']) == 0:
        f_handle = open(os.path.join(save_folder, 'log.log'), 'w', buffering=1)
    else:
        clean_log_before_continuing(os.path.join(save_folder, 'log.log'), start_iter)
        f_handle = open(os.path.join(save_folder, 'log.log'), 'a', buffering=1)

    val_iter = 0
    curr_iter = start_iter
    while curr_iter <= args['max_iter']:
        if not skip_clustering:
            # Extract image features from reference images
            net.eval()
            net.output_features = True

            features, _ = extract_features_for_reference(net, model_config, ref_image_lists,
                                                         corr_im_paths, ref_featurs_pos,
                                                         max_num_features_per_image=args['max_features_per_image'],
                                                         fraction_correspondeces=0.5)

            cluster_features = np.vstack(features)
            del features

            # cluster the features
            cluster_indices, clustering_loss, cluster_centroids, pca_info = deepcluster.cluster_imfeatures(
                cluster_features, verbose=True, use_gpu=False)

            # save cluster centroids
            h5f = h5py.File(os.path.join(save_folder, 'centroids_%d.h5' % curr_iter), 'w')
            h5f.create_dataset('cluster_centroids', data=cluster_centroids)
            h5f.create_dataset('pca_transform_Amat', data=pca_info[0])
            h5f.create_dataset('pca_transform_bvec', data=pca_info[1])
            h5f.close()

            # Print distribution of clusters
            cluster_distribution, _ = np.histogram(
                cluster_indices, bins=np.arange(args['n_clusters'] + 1), density=True)
            str2write = 'cluster distribution ' + \
                np.array2string(cluster_distribution, formatter={'float_kind': '{0:.8f}'.format}).replace('\n', ' ')
            print(str2write)
            f_handle.write(str2write + "\n")

            reinit_last_layers(net)  # set last layer weight to a normal distribution

            # make a copy of current network state to do cluster assignment
            net_for_clustering = copy.deepcopy(net)
        else:
            skip_clustering = False

        optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['max_iter']
                                                            ) ** args['lr_decay']
        optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['max_iter']
                                                        ) ** args['lr_decay']

        net.train()
        freeze_bn(net)
        net.output_features = False
        cluster_training_count = 0

        # Train using the training correspondence set
        corr_train_loss = AverageMeter()
        seg_train_loss = AverageMeter()
        feature_train_loss = AverageMeter()

        while cluster_training_count < args['cluster_interval'] and curr_iter <= args['max_iter']:

            # First extract cluster labels using saved network checkpoint
            net.to("cpu")
            net_for_clustering.to(device)
            net_for_clustering.eval()
            net_for_clustering.output_features = True
            if args['feature_hinge_loss_weight'] > 0:
                net_for_clustering.output_all = False
            data_samples = []
            extract_label_count = 0
            while (extract_label_count < args['chunk_size']) and (cluster_training_count + extract_label_count < args['cluster_interval']) and (val_iter + extract_label_count < args['val_interval']) and (extract_label_count + curr_iter <= args['max_iter']):
                img_ref, img_other, pts_ref, pts_other, _ = next(iter(corr_loader_train))

                # Transfer data to device
                img_ref = img_ref.to(device)
                img_other = img_other.to(device)
                pts_ref = [p.to(device) for p in pts_ref]
                pts_other = [p.to(device) for p in pts_other]

                with torch.no_grad():
                    features = net_for_clustering(img_ref)

                # assign feature to clusters for entire patch
                output = features.cpu().numpy()
                output_flat = output.reshape((output.shape[0], output.shape[1], -1))
                cluster_image = np.zeros((output.shape[0], output.shape[2], output.shape[3]), dtype=np.int64)
                for b in range(output_flat.shape[0]):
                    out_f = output_flat[b]
                    out_f2, _ = preprocess_features(np.swapaxes(out_f, 0, 1), pca_info=pca_info)
                    cluster_labels = deepcluster.assign(out_f2)
                    cluster_image[b] = cluster_labels.reshape((output.shape[2], output.shape[3]))

                cluster_image = torch.from_numpy(cluster_image).to(device)

                # assign cluster to correspondence positions
                cluster_labels = assign_cluster_ids_to_correspondence_points(
                    features, pts_ref, (deepcluster, pca_info), inds_other=pts_other, orig_im_size=(713, 713))

                # Transfer data to cpu
                img_ref = img_ref.cpu()
                img_other = img_other.cpu()
                pts_ref = [p.cpu() for p in pts_ref]
                pts_other = [p.cpu() for p in pts_other]
                cluster_labels = [p.cpu() for p in cluster_labels]
                cluster_image = cluster_image.cpu()
                data_samples.append((img_ref, img_other, pts_ref, pts_other, cluster_labels, cluster_image))
                extract_label_count += 1

            net_for_clustering.to("cpu")
            net.to(device)

            for data_sample in data_samples:
                img_ref, img_other, pts_ref, pts_other, cluster_labels, cluster_image = data_sample

                # Transfer data to device
                img_ref = img_ref.to(device)
                img_other = img_other.to(device)
                pts_ref = [p.to(device) for p in pts_ref]
                pts_other = [p.to(device) for p in pts_other]
                cluster_labels = [p.to(device) for p in cluster_labels]
                cluster_image = cluster_image.to(device)

                optimizer.zero_grad()

                if args['feature_hinge_loss_weight'] > 0:
                    net.output_all = True

                # Randomization to decide if reference or target image should be used for training
                if args['fraction_reference_bp'] is None:  # use both
                    if args['feature_hinge_loss_weight'] > 0:
                        out_feat_ref, aux_feat_ref, outputs_ref, aux_ref = net(img_ref)
                    else:
                        outputs_ref, aux_ref = net(img_ref)

                    seg_main_loss = seg_loss_fct(outputs_ref, cluster_image)
                    seg_aux_loss = seg_loss_fct(aux_ref, cluster_image)

                    if args['feature_hinge_loss_weight'] > 0:
                        out_feat_other, aux_feat_other, outputs_other, aux_other = net(img_other)
                    else:
                        outputs_other, aux_other = net(img_other)

                elif np.random.rand(1)[0] < args['fraction_reference_bp']:  # use reference
                    if args['feature_hinge_loss_weight'] > 0:
                        out_feat_ref, aux_feat_ref, outputs_ref, aux_ref = net(img_ref)
                    else:
                        outputs_ref, aux_ref = net(img_ref)

                    seg_main_loss = seg_loss_fct(outputs_ref, cluster_image)
                    seg_aux_loss = seg_loss_fct(aux_ref, cluster_image)

                    with torch.no_grad():
                        if args['feature_hinge_loss_weight'] > 0:
                            out_feat_other, aux_feat_other, outputs_other, aux_other = net(img_other)
                        else:
                            outputs_other, aux_other = net(img_other)
                else:  # use target
                    with torch.no_grad():
                        if args['feature_hinge_loss_weight'] > 0:
                            out_feat_ref, aux_feat_ref, outputs_ref, aux_ref = net(img_ref)
                        else:
                            outputs_ref, aux_ref = net(img_ref)

                    if args['feature_hinge_loss_weight'] > 0:
                        out_feat_other, aux_feat_other, outputs_other, aux_other = net(img_other)
                    else:
                        outputs_other, aux_other = net(img_other)

                    seg_main_loss = 0.
                    seg_aux_loss = 0.

                if args['feature_hinge_loss_weight'] > 0:
                    net.output_all = False

                main_loss, _ = loss_fct(outputs_ref, outputs_other, None, pts_ref,
                                        pts_other, cluster_labels=cluster_labels)
                aux_loss, _ = loss_fct(aux_ref, aux_other, None, pts_ref, pts_other, cluster_labels=cluster_labels)

                if args['feature_hinge_loss_weight'] > 0:
                    feature_loss = feature_loss_fct(out_feat_ref, out_feat_other, pts_ref, pts_other, None)
                    feature_loss_aux = feature_loss_fct(aux_feat_ref, aux_feat_other, pts_ref, pts_other, None)
                    loss = args['corr_loss_weight']*(main_loss + 0.4 * aux_loss) + args['seg_loss_weight']*(seg_main_loss + 0.4 * seg_aux_loss) \
                        + args['feature_hinge_loss_weight']*(feature_loss + 0.4 * feature_loss_aux)
                else:
                    feature_loss = 0.
                    loss = args['corr_loss_weight']*(main_loss + 0.4 * aux_loss) + \
                        args['seg_loss_weight']*(seg_main_loss + 0.4 * seg_aux_loss)

                loss.backward()
                optimizer.step()
                cluster_training_count += 1

                corr_train_loss.update(main_loss.item(), 1)
                if type(seg_main_loss) == torch.Tensor:
                    seg_train_loss.update(seg_main_loss.item(), 1)
                if type(feature_loss) == torch.Tensor:
                    feature_train_loss.update(feature_loss.item(), 1)

                ####################################################################################################
                #       LOGGING ETC
                ####################################################################################################
                curr_iter += 1
                val_iter += 1

                writer.add_scalar('train_corr_loss', corr_train_loss.avg, curr_iter)
                writer.add_scalar('train_seg_loss', seg_train_loss.avg, curr_iter)
                writer.add_scalar('train_feature_loss', feature_train_loss.avg, curr_iter)
                writer.add_scalar('lr', optimizer.param_groups[1]['lr'], curr_iter)

                if (curr_iter + 1) % args['print_freq'] == 0:
                    str2write = '[iter %d / %d], [train seg loss %.5f], [train corr loss %.5f], [train feature loss %.5f]. [lr %.10f]' % (
                        curr_iter+1, args['max_iter'], seg_train_loss.avg, corr_train_loss.avg,
                        feature_train_loss.avg, optimizer.param_groups[1]['lr'])

                    print(str2write)
                    f_handle.write(str2write + "\n")

                if val_iter >= args['val_interval']:
                    val_iter = 0
                    net_for_clustering.to(device)
                    corr_validator.run(net, net_for_clustering, (deepcluster, pca_info), optimizer,
                                       args, curr_iter, save_folder, f_handle, writer=writer)
                    net_for_clustering.to("cpu")

                if curr_iter > args['max_iter']:
                    break

    # Post training
    f_handle.close()
    writer.close()


def generate_name_of_result_folder(args):
    global_opts = get_global_opts()

    results_path = os.path.join(global_opts['result_path'], 'cluster-training')
    if 'vis' == args['startnet']:
        startnetstr = 'map1'
    elif 'cs' == args['startnet']:
        startnetstr = 'map0'
    else:
        startnetstr = 'other'

    cluster_str = 'features%d' % (args['max_features_per_image'])

    if args['feature_hinge_loss_weight'] == 0:
        result_folder = 'cluster-%s-%s-cn%d-ci%d-vi%d-wc%.5f-ws%.5f-%s-valm-%s-%.10flr' % (
            args['corr_set'], startnetstr, args['n_clusters'], args['cluster_interval'],
            args['val_interval'], args['corr_loss_weight'], args['seg_loss_weight'],
            cluster_str, args['feature_distance_measure'], args['lr'])
    else:
        result_folder = 'cluster-%s-%s-cn%d-ci%d-vi%d-wc%.5f-ws%.5f-wf%.5f-%s-valm-%s-%.10flr' % (
            args['corr_set'], startnetstr, args['n_clusters'], args['cluster_interval'],
            args['val_interval'], args['corr_loss_weight'], args['seg_loss_weight'],
            args['feature_hinge_loss_weight'], cluster_str, args['feature_distance_measure'], args['lr'])

    return os.path.join(results_path, result_folder), os.path.join(global_opts['cache_path'], result_folder)


def get_path_of_startnet(args):
    global_opts = get_global_opts()

    if args['startnet'] == 'vis':
        return os.path.join(global_opts['result_path'], 'base-networks', 'pspnet101_cs_vis.pth')
    elif args['startnet'] == 'cs':
        return os.path.join(global_opts['result_path'], 'base-networks', 'pspnet101_cityscapes.pth')


def train_with_clustering_experiment(args):
    if args['startnet'] in ['vis', 'cs']:
        startnet = get_path_of_startnet(args)
    else:
        startnet = args['startnet']

    save_folder, tmp_folder = generate_name_of_result_folder(args)
    train_with_clustering(save_folder, tmp_folder, startnet, args)


if __name__ == '__main__':
    args = {
        # general training settings
        'train_batch_size': 1,
        # probability of propagating error for reference image instead of target imare (set to None to use both)
        'fraction_reference_bp': 0.5,
        'lr': 1e-4 / sqrt(16 / 1),
        'lr_decay': 1,
        'max_iter': 60000,
        'weight_decay': 1e-4,
        'momentum': 0.9,

        # starting network settings
        'startnet': 'vis',  # specify full path or set to 'vis' for network trained with vistas + cityscapes or 'cs' for network trained with cityscapes
        'use_original_base': False,  # must be true if starting from classification network

        # set to '' to start training from beginning and 'latest' to use last checkpoint
        'snapshot': 'latest',

        # dataset settings
        'corr_set': 'cmu',  # 'cmu', 'rc', 'both' or 'none'
        'max_features_per_image': 500,  # dont set to high (RAM runs out)

        # clustering settings
        'n_clusters': 100,
        'cluster_interval': 10000,

        # loss settings
        'corr_loss_weight': 1,  # was 1
        'seg_loss_weight': 1,  # was 1
        'feature_hinge_loss_weight': 0,  # was 0

        # validation settings
        'val_interval': 2500,
        'feature_distance_measure': 'L2',

        # misc
        'chunk_size': 50,
        'print_freq': 10,
        'stride_rate': 2 / 3.,
        'n_workers': 1,  # set to 0 for debugging
    }
    train_with_clustering_experiment(args)
