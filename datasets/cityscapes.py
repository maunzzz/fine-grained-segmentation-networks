import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from utils.misc import remap_mask

num_classes = 19
ignore_label = 255

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32, 56, 165, 134]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def make_dataset(im_folder, seg_folder, im_file_ending, seg_file_ending):
    items = list()
    for root, subdirs, files in os.walk(im_folder):
        for ff in files:
            if ff.endswith(im_file_ending):
                # Create file name for segmentation
                seg_path = os.path.join(seg_folder, root.replace(im_folder, '').strip('/'),
                                        ff.replace(im_file_ending, seg_file_ending))
                # If segmentation exists, add it to list of files
                if os.path.isfile(seg_path):
                    items.append((os.path.join(root, ff), seg_path))
    return items


class CityScapes(data.Dataset):
    def __init__(self, im_folder, seg_folder, im_file_ending, seg_file_ending='png', id_to_trainid=None, joint_transform=None, sliding_crop=None, transform=None, target_transform=None, transform_before_sliding=None):
        self.imgs = make_dataset(im_folder, seg_folder, im_file_ending, seg_file_ending)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.transform_before_sliding = transform_before_sliding
        self.id_to_trainid = id_to_trainid

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            # from 0,1,2,...,255 to 0,1,2,3,... (to set introduced pixels due to transform to ignore)
            mask = remap_mask(mask, 0)
            img, mask = self.joint_transform(img, mask)
            mask = remap_mask(mask, 1)  # back again

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.sliding_crop is not None:
            if self.transform_before_sliding is not None:
                img = self.transform_before_sliding(img)
            img_slices, slices_info = self.sliding_crop(img)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            img = torch.stack(img_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            return img, mask

    def __len__(self):
        return len(self.imgs)
