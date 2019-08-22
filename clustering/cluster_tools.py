from models import model_configs
import utils.joint_transforms as joint_transforms
from utils.misc import get_global_opts, im_to_ext_name
from clustering.clustering import preprocess_features
import os
import sys
import numpy as np
import h5py
import torch
from PIL import Image
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def create_interpol_weights(wsize, sliding_transform_step):
    interpol_weight = torch.zeros(wsize)
    interpol_weight += 1.0

    grade_length_x = round(wsize[1]*(1-sliding_transform_step[0]))
    grade_length_y = round(wsize[0]*(1-sliding_transform_step[1]))

    for k in range(grade_length_x):
        interpol_weight[:, k] *= (k+1)/grade_length_x
        interpol_weight[:, -(k+1)] *= (k+1)/grade_length_x

    for k in range(grade_length_y):
        interpol_weight[k, :] *= (k+1)/grade_length_y
        interpol_weight[-(k+1), :] *= (k+1)/grade_length_y

    return interpol_weight


def extract_features_for_reference_nocorr(net, net_config, train_set, num_images, max_num_features_per_image=500):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    features = []
    t0 = time.time()

    if num_images < len(train_set):
        sample_ids = np.random.choice(len(train_set), num_images, replace=False)
    else:
        sample_ids = list(range(len(train_set)))

    for counter, iii in enumerate(sample_ids):
        img_slices, mask, slices_info = train_set[iii]

        # run network on all slizes
        img_slices = img_slices.to(device)

        for ind in range(0, img_slices.size(0), 10):
            max_ind = min(ind + 10, img_slices.size(0))
            with torch.no_grad():
                out_tmp = net(img_slices[ind:max_ind, :, :, :])
            if ind == 0:
                n_features = out_tmp.size(1)
                oh = out_tmp.size(2)
                ow = out_tmp.size(3)
                scale_h = 713/oh
                scale_w = 713/ow
                output_slices = torch.zeros(img_slices.size(0), n_features, oh, ow).to(device)

            output_slices[ind:max_ind] = out_tmp

        # merge to one image
        outsizeh = round(slices_info[:, 0].max().item()/scale_h) + oh
        outsizew = round(slices_info[:, 2].max().item()/scale_w) + ow
        count = torch.zeros(outsizeh, outsizew).to(device)
        output = torch.zeros(n_features, outsizeh, outsizew).to(device)
        sliding_transform_step = (2/3, 2/3)
        interpol_weight = create_interpol_weights((oh, ow), sliding_transform_step)
        interpol_weight = interpol_weight.to(device)

        for output_slice, info in zip(output_slices, slices_info):
            hs = round(info[0].item()/scale_h)
            ws = round(info[2].item()/scale_w)
            output[:, hs:hs+oh, ws:ws + ow] += (interpol_weight*output_slice[:, :oh, :ow]).data
            count[hs:hs+oh, ws:ws+ow] += interpol_weight

        output /= count

        if max_num_features_per_image > output.size(1)*output.size(2):
            for indi in range(output.size(1)):
                for indj in range(output.size(2)):
                    features.append(output[:, indi, indj].cpu().numpy().astype(np.float32))
        else:
            # add random points
            lin_inds = np.random.choice(output.size(2)*output.size(1), max_num_features_per_image, replace=False)
            pts = np.concatenate((np.expand_dims(lin_inds // output.size(1), axis=0),
                                  np.expand_dims(lin_inds % output.size(1), axis=0)), axis=0)
            for pti in range(pts.shape[1]):
                features.append(output[:, pts[1, pti], pts[0, pti]].cpu().numpy().astype(np.float32))

        if (counter % 100) == 0:
            print('computing features: %d/%d' % (counter, len(sample_ids)))

    tend = time.time()
    print('Time: %f' % (tend-t0))

    return features

# see below how to call this function


def extract_features_for_reference(net, net_config, im_list_file, im_root, ref_pts_dir, max_num_features_per_image=500, fraction_correspondeces=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    features = []

    # store mapping from original image representation to linear feature index
    # for correspondence coordinate with index i in image with index f
    # feature_ind = mapping[f][i][0]
    # x_coord = mapping[f][i][1]
    # y_coord = mapping[f][i][2]
    #
    # features[feature_ind] was taken from output_feature_map[:, y_coord, x_coord] of image f
    mapping = []

    # data loading
    input_transform = net_config.input_transform
    pre_inference_transform_with_corrs = net_config.pre_inference_transform_with_corrs
    # make sure crop size and stride same as during training
    sliding_crop = joint_transforms.SlidingCropImageOnly(713, 2/3)

    t0 = time.time()

    # get all file names

    filenames_ims = list()
    filenames_pts = list()
    for im_l, im_r, ref_pts_d in zip(im_list_file, im_root, ref_pts_dir):
        with open(im_l) as f:
            for line in f:
                filename_im = line.strip()

                filenames_ims.append(os.path.join(im_r, filename_im))
                filenames_pts.append(os.path.join(
                    ref_pts_d, im_to_ext_name(filename_im, 'h5')))

    # filenames_ims = [filenames_ims[0]]
    # filenames_pts = [filenames_pts[0]]

    iii = 0
    feature_count = 0
    for filename_im, filename_pts in zip(filenames_ims, filenames_pts):

        img = Image.open(filename_im).convert('RGB')
        f = h5py.File(filename_pts, 'r')
        pts = np.swapaxes(np.array(f['corr_pts']), 0, 1)

        # creating sliding crop windows and transform them
        if pre_inference_transform_with_corrs is not None:
            img, pts = pre_inference_transform_with_corrs(img, pts)

        img_size = img.size
        img_slices, slices_info = sliding_crop(img)
        if input_transform is not None:
            img_slices = [input_transform(e) for e in img_slices]

        img_slices = torch.stack(img_slices, 0)
        slices_info = torch.LongTensor(slices_info)
        slices_info.squeeze_(0)

        # run network on all slizes
        img_slices = img_slices.to(device)

        for ind in range(0, img_slices.size(0), 10):
            max_ind = min(ind + 10, img_slices.size(0))
            with torch.no_grad():
                out_tmp = net(img_slices[ind:max_ind, :, :, :])
            if ind == 0:
                n_features = out_tmp.size(1)
                oh = out_tmp.size(2)
                ow = out_tmp.size(3)
                scale_h = 713/oh
                scale_w = 713/ow
                output_slices = torch.zeros(
                    img_slices.size(0), n_features, oh, ow).to(device)
            output_slices[ind:max_ind] = out_tmp

        outsizeh = round(slices_info[:, 0].max().item()/scale_h) + oh
        outsizew = round(slices_info[:, 2].max().item()/scale_w) + ow
        count = torch.zeros(outsizeh, outsizew).to(device)
        output = torch.zeros(n_features, outsizeh, outsizew).to(device)
        sliding_transform_step = (2/3, 2/3)
        interpol_weight = create_interpol_weights(
            (oh, ow), sliding_transform_step)
        interpol_weight = interpol_weight.to(device)

        for output_slice, info in zip(output_slices, slices_info):
            hs = round(info[0].item()/scale_h)
            ws = round(info[2].item()/scale_w)
            output[:, hs:hs+oh, ws:ws +
                   ow] += (interpol_weight*output_slice[:, :oh, :ow]).data
            count[hs:hs+oh, ws:ws+ow] += interpol_weight
        output /= count

        # Scale correspondences coordinates to output size
        pts[0, :] = pts[0, :]*outsizew/img_size[0]
        pts[1, :] = pts[1, :]*outsizeh/img_size[1]
        pts = (pts + 0.5).astype(int)
        valid_inds = (pts[0, :] < output.size(2)) & (pts[0, :] >= 0) & (
            pts[1, :] < output.size(1)) & (pts[1, :] >= 0)
        pts = pts[:, valid_inds]
        pts = np.unique(pts, axis=1)

        if pts.shape[1] > int(max_num_features_per_image*fraction_correspondeces + 0.5):
            inds = np.random.choice(pts.shape[1], int(
                max_num_features_per_image*fraction_correspondeces + 0.5), replace=False)
            pts = pts[:, inds]

        # add some random points as well
        n_non_corr = int((1 - fraction_correspondeces) /
                         fraction_correspondeces * pts.shape[1] + 0.5)
        lin_inds = np.random.choice(output.size(
            2)*output.size(1), n_non_corr, replace=False)
        ptstoadd = np.concatenate((np.expand_dims(
            lin_inds // output.size(1), axis=0), np.expand_dims(lin_inds % output.size(1), axis=0)), axis=0)
        pts = np.hstack((pts, ptstoadd))

        # Save features
        thismap = {}
        thismap['coords'] = []
        thismap['size'] = (outsizeh, outsizew)
        for pti in range(pts.shape[1]):
            features.append(
                output[:, pts[1, pti], pts[0, pti]].cpu().numpy().astype(np.float32))
            thismap['coords'].append([feature_count, pts[0, pti], pts[1, pti]])
            feature_count += 1

        mapping.append(thismap)

        if (iii % 100) == 0:
            print('computing features: %d/%d' % (iii, len(filenames_ims)))

        iii += 1

    tend = time.time()
    print('Time: %f' % (tend-t0))

    return features, mapping


def assign_cluster_ids_to_correspondence_points(features, inds, cluster_info, inds_other=None, orig_im_size=(713, 713)):
    features_flat = features.view([features.size(0), features.size(1), -1])
    cluster_labels = []
    for b in range(features.size(0)):
        b_inds = inds[b]

        if orig_im_size != (features.size(2), features.size(3)):
            b_inds = b_inds.type(torch.float32)
            b_inds[0, :] = b_inds[0, :]/orig_im_size[1]*features.size(3)
            b_inds[1, :] = b_inds[1, :]/orig_im_size[0]*features.size(2)

        b_inds = (b_inds + .5).type(torch.int64)

        if inds_other is not None:
            b_inds_other = inds_other[b]

            if orig_im_size != (features.size(2), features.size(3)):
                b_inds_other = b_inds_other.type(torch.float32)
                b_inds_other[0, :] = b_inds_other[0, :] / \
                    orig_im_size[1]*features.size(3)
                b_inds_other[1, :] = b_inds_other[1, :] / \
                    orig_im_size[0]*features.size(2)

            b_inds_other = (b_inds_other + .5).type(torch.int64)
            valid_inds = (b_inds[0, :] < features.size(3)) & (b_inds[0, :] >= 0) & (b_inds[1, :] < features.size(2)) & (b_inds[1, :] >= 0) & (
                b_inds_other[0, :] < features.size(3)) & (b_inds_other[0, :] >= 0) & (b_inds_other[1, :] < features.size(2)) & (b_inds_other[1, :] >= 0)
        else:
            valid_inds = (b_inds[0, :] < features.size(3)) & (b_inds[0, :] >= 0) & (
                b_inds[1, :] < features.size(2)) & (b_inds[1, :] >= 0)

        lin_b_inds = features.size(2)*b_inds[1, :] + b_inds[0, :]
        lin_b_inds = lin_b_inds[valid_inds]

        # This could probably be done a lot nicer
        numpy_features = np.transpose(
            features_flat[b, :, lin_b_inds].clone().detach().cpu().numpy())
        # PCA-reducing, whitening and L2-normalization
        numpy_features, _ = preprocess_features(
            numpy_features, pca_info=cluster_info[1])
        cluster_labels_b = cluster_info[0].assign(numpy_features)
        cluster_labels_b = np.squeeze(cluster_labels_b)

        cluster_labels_b = torch.from_numpy(
            cluster_labels_b).unsqueeze(0).to(features.device)
        cluster_labels.append(cluster_labels_b)

    return cluster_labels


def save_cluster_features_as_segmentations(list_file, tmp_folder, im_mappings, cluster_indices):
    filenames_ims = []
    filenames_segs = []
    with open(list_file) as f:
        for line in f:
            im_name = line.strip()
            filenames_ims.append(im_name)
            filenames_segs.append(im_to_ext_name(im_name, 'png'))

    for mapping, save_name in zip(im_mappings, filenames_segs):
        cluster_map = 255*np.ones(mapping['size'], dtype=np.int32)
        for cc in mapping['coords']:
            cluster_map[cc[2], cc[1]] = cluster_indices[cc[0]]

        new_mask = Image.fromarray(cluster_map.astype(np.uint8)).convert('P')
        new_mask.save(os.path.join(tmp_folder, save_name))

    return filenames_ims, filenames_segs
