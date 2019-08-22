import os
import random

import numpy as np
import h5py
import torch
from PIL import Image, ImageOps
from torch.utils import data


def correspondences_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size, except for correspondence points which will be a list of tensors"""

    ims1 = torch.stack([b[0] for b in batch], 0)
    ims2 = torch.stack([b[1] for b in batch], 0)
    pts1 = [torch.from_numpy(b[2]) for b in batch]
    pts2 = [torch.from_numpy(b[3]) for b in batch]
    weights = [torch.from_numpy(b[4]) for b in batch]
    return [ims1, ims2, pts1, pts2, weights]


def refine_correspondence_sample(ref_probmaps, other_probmaps, ref_inds, other_inds, weights, remove_same_class=True, remove_classes=None):
    probmap_ref_flat = ref_probmaps.view(
        [ref_probmaps.size(0), ref_probmaps.size(1), -1])
    probmap_other_flat = other_probmaps.view(
        [other_probmaps.size(0), other_probmaps.size(1), -1])

    ref_inds_out = list()
    other_inds_out = list()
    weights_out = list()
    batch_inds_to_keep = torch.zeros(
        [len(ref_inds)], dtype=torch.uint8, device=ref_inds[0].device)
    for b in range(ref_probmaps.size(0)):
        # get variables for this batch
        inds_ref_this = ref_inds[b].type(torch.int64)
        inds_other_this = other_inds[b].type(torch.int64)
        weights_this = weights[b]

        # convert to linear indices
        lin_inds_ref = ref_probmaps.size(
            2)*inds_ref_this[1, :] + inds_ref_this[0, :]
        lin_inds_other = other_probmaps.size(
            2)*inds_other_this[1, :] + inds_other_this[0, :]

        # get predictions
        ref_pred_flat = probmap_ref_flat[b, :, lin_inds_ref].max(0)[
            1].squeeze_(0)
        other_pred_flat = probmap_other_flat[b, :, lin_inds_other].max(0)[
            1].squeeze_(0)

        inds_to_keep = ref_pred_flat >= 0

        # remove samples with non-stationary classes
        if remove_classes is not None:
            for class_to_remove in remove_classes:
                inds_to_keep = inds_to_keep - \
                    (ref_pred_flat == class_to_remove)

        # remove samples where correspondences have the same class
        if remove_same_class:
            inds_to_keep = torch.max(inds_to_keep - (ref_pred_flat == other_pred_flat),
                                     torch.tensor(0, dtype=inds_to_keep.dtype, device=inds_to_keep.device))

        # append correspondence that should
        ref_inds_out.append(inds_ref_this[:, inds_to_keep])
        other_inds_out.append(inds_other_this[:, inds_to_keep])
        weights_out.append(weights_this[inds_to_keep, :])

        if inds_to_keep.sum() > 0:
            batch_inds_to_keep[b] = 1

    return ref_inds_out, other_inds_out, weights_out, batch_inds_to_keep


def make_dataset(corr_path, listfile=None):
    items = []
    if listfile is None:
        f_name_list = [fn for fn in os.listdir(
            corr_path) if fn.endswith('mat')]
    else:
        f_name_list = []
        with open(listfile) as f:
            for line in f:
                f_name_list.append(line.strip())

    for f_name in f_name_list:
        item = (os.path.join(corr_path, f_name))
        items.append(item)

    return items


class Correspondences(data.Dataset):
    def __init__(self, corr_path, im_path, input_size=(713, 713), input_transform=None, joint_transform=None, listfile=None):
        self.data = make_dataset(corr_path, listfile)
        if len(self.data) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.input_size = input_size
        self.root_imgs = im_path
        self.transform = input_transform
        self.joint_transform = joint_transform

    def __getitem__(self, index):
        # Load data from one sample
        mat_content = {}
        f = h5py.File(self.data[index], 'r')
        for k, v in f.items():
            mat_content[k] = np.array(v)

        im1name = ''.join(chr(a)
                          for a in mat_content['im_i_path'])  # convert to string
        im2name = ''.join(chr(a)
                          for a in mat_content['im_j_path'])  # convert to string
        mat_content['pt_i'] = np.swapaxes(mat_content['pt_i'], 0, 1)
        mat_content['pt_j'] = np.swapaxes(mat_content['pt_j'], 0, 1)
        mat_content['dist_from_center'] = np.swapaxes(
            mat_content['dist_from_center'], 0, 1)

        img1path = os.path.join(self.root_imgs, im1name)
        img2path = os.path.join(self.root_imgs, im2name)

        img1 = Image.open(img1path).convert('RGB')
        img2 = Image.open(img2path).convert('RGB')
        pts1 = mat_content['pt_i']
        pts2 = mat_content['pt_j']
        dists = mat_content['dist_from_center']
        weights = np.exp((-1*dists))

        if self.joint_transform is not None:
            img1, img2, pts1, pts2, weights = self.joint_transform(
                img1, img2, pts1, pts2, weights)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, (pts1 + .5).astype(np.int32), (pts2 + .5).astype(np.int32), weights.astype(np.float32)

    def __len__(self):
        return len(self.data)
