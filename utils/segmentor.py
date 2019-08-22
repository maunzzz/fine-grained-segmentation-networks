import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as standard_transforms
from torch.autograd import Variable
import numbers
import torchvision.transforms.functional as F
import h5py
import scipy.io as sio
import sys
import context
from utils.misc import check_mkdir
import utils.joint_transforms as joint_transforms


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


class Segmentor():
    def __init__(
        self,
        network,
        num_classes,
        n_slices_per_pass=5,
        colorize_fcn=None,
    ):
        self.net = network
        self.num_classes = num_classes
        self.colorize_fcn = colorize_fcn
        self.n_slices_per_pass = n_slices_per_pass

    def run_on_slices(self, img_slices, slices_info, sliding_transform_step=2/3., use_gpu=True, return_logits=False):
        if isinstance(sliding_transform_step, numbers.Number):
            sliding_transform_step = (sliding_transform_step, sliding_transform_step)

        imsize1 = slices_info[:, 1].max().item()
        imsize2 = slices_info[:, 3].max().item()

        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = "cpu"

        interpol_weight = create_interpol_weights((img_slices.size(2), img_slices.size(3)), sliding_transform_step)
        #interpol_weight = interpol_weight.to(device)

        #count = torch.zeros(imsize1, imsize2).to(device)
        #output = torch.zeros(self.num_classes, imsize1, imsize2).to(device)
        count = torch.zeros(imsize1, imsize2)
        output = torch.zeros(self.num_classes, imsize1, imsize2)

        # run network on all slizes
        img_slices = img_slices.to(device)
        output_slices = torch.zeros(img_slices.size(
            0), self.num_classes, img_slices.size(2), img_slices.size(3))
        # output_slices = torch.zeros(img_slices.size(0), self.num_classes, img_slices.size(2), img_slices.size(3)).to(device)
        for ind in range(0, img_slices.size(0), self.n_slices_per_pass):
            max_ind = min(ind + self.n_slices_per_pass, img_slices.size(0))
            with torch.no_grad():
                #output_slices[ind:max_ind, :, :, :] = self.net(img_slices[ind:max_ind, :, :, :])
                output_slices[ind:max_ind, :, :, :] = self.net(img_slices[ind:max_ind, :, :, :]).cpu()

        for output_slice, info in zip(output_slices, slices_info):
            weighted_output = (interpol_weight*output_slice)
            # weighted_output = (interpol_weight*output_slice).cpu()
            output[:, info[0]: info[1], info[2]: info[3]] += weighted_output[:, :info[4], :info[5]]
            #count[info[0]: info[1], info[2]: info[3]] += (interpol_weight[:info[4], :info[5]]).cpu()
            count[info[0]: info[1], info[2]: info[3]] += (interpol_weight[:info[4], :info[5]])

        output /= count
        del img_slices
        del output_slices
        del interpol_weight
        del weighted_output

        if return_logits:
            return output.cpu()
        else:
            return output.max(0)[1].squeeze_(0).cpu().numpy()

    def run_and_save(
        self,
        img_path,
        seg_path,
        pre_sliding_crop_transform=None,
        sliding_crop=joint_transforms.SlidingCropImageOnly(713, 2 / 3.),
        input_transform=standard_transforms.ToTensor(),
        verbose=False,
        skip_if_seg_exists=False,
        use_gpu=True,
    ):
        """
        img                  - Path of input image
        seg_path             - Path of output image (segmentation)
        sliding_crop         - Transform that returns set of image slices
        input_transform      - Transform to apply to image before inputting to network
        skip_if_seg_exists   - Whether to overwrite or skip if segmentation exists already
        """

        if seg_path is not None:
            if os.path.exists(seg_path):
                if skip_if_seg_exists:
                    if verbose:
                        print(
                            "Segmentation already exists, skipping: {}".format(seg_path))
                    return
                else:
                    if verbose:
                        print(
                            "Segmentation already exists, overwriting: {}".format(seg_path))

        if isinstance(img_path, str):
            try:
                img = Image.open(img_path).convert('RGB')
            except OSError:
                print("Error reading input image, skipping: {}".format(img_path))
                return
        else:
            img = img_path

        # creating sliding crop windows and transform them
        img_size_orig = img.size
        if pre_sliding_crop_transform is not None:  # might reshape image
            img = pre_sliding_crop_transform(img)

        img_slices, slices_info = sliding_crop(img)
        img_slices = [input_transform(e) for e in img_slices]
        img_slices = torch.stack(img_slices, 0)
        slices_info = torch.LongTensor(slices_info)
        slices_info.squeeze_(0)

        prediction_logits = self.run_on_slices(
            img_slices, slices_info, sliding_transform_step=sliding_crop.stride_rate, use_gpu=use_gpu, return_logits=True)
        prediction_orig = prediction_logits.max(0)[1].squeeze_(0).numpy()
        prediction_logits = prediction_logits.numpy()

        if self.colorize_fcn:
            prediction_colorized = self.colorize_fcn(prediction_orig)
        else:
            prediction_colorized = Image.fromarray(
                prediction_orig.astype(np.int32)).convert('I')

        if prediction_colorized.size != img_size_orig:
            prediction_colorized = F.resize(
                prediction_colorized, img_size_orig[::-1], interpolation=Image.NEAREST)

        if seg_path is not None:
            check_mkdir(os.path.dirname(seg_path))
            prediction_colorized.save(seg_path)

        return prediction_colorized


class FeatureExtractor():
    def __init__(
        self,
        network,
        n_slices_per_pass=5
    ):
        self.net = network
        self.n_slices_per_pass = n_slices_per_pass

    def run_on_slices(self, img_slices, slices_info, sliding_transform_step=2/3., use_gpu=True):
        if isinstance(sliding_transform_step, numbers.Number):
            sliding_transform_step = (sliding_transform_step, sliding_transform_step)

        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = "cpu"

        interpol_weight = create_interpol_weights((img_slices.size(2), img_slices.size(3)), sliding_transform_step)

        # run network on all slizes
        img_slices = img_slices.to(device)
        # output_slices = torch.zeros(img_slices.size(0), self.num_classes, img_slices.size(2), img_slices.size(3)).to(device)
        for ind in range(0, img_slices.size(0), self.n_slices_per_pass):
            max_ind = min(ind + self.n_slices_per_pass, img_slices.size(0))
            with torch.no_grad():
                out_tmp = self.net(img_slices[ind:max_ind, :, :, :])
            if ind == 0:
                dim_features = out_tmp.size(1)
                oh = out_tmp.size(2)
                ow = out_tmp.size(3)
                scale_h = img_slices.size(2)/oh
                scale_w = img_slices.size(3)/ow
                output_slices = torch.zeros(
                    img_slices.size(0), dim_features, oh, ow).to(device)
            output_slices[ind:max_ind] = out_tmp

        outsizeh = round(slices_info[:, 0].max().item()/scale_h) + oh
        outsizew = round(slices_info[:, 2].max().item()/scale_w) + ow
        count = torch.zeros(outsizeh, outsizew).to(device)
        output = torch.zeros(dim_features, outsizeh, outsizew).to(device)
        sliding_transform_step = (2/3, 2/3)
        interpol_weight = create_interpol_weights(
            (oh, ow), sliding_transform_step)
        interpol_weight = interpol_weight.to(device)

        for output_slice, info in zip(output_slices, slices_info):
            hs = round(info[0].item()/scale_h)
            ws = round(info[2].item()/scale_w)
            output[:, hs:hs+oh, ws:ws + ow] += (interpol_weight*output_slice[:, :oh, :ow]).data
            count[hs:hs+oh, ws:ws+ow] += interpol_weight
        output /= count

        return output.cpu().numpy()

    def run_and_save(
        self,
        img_path,
        save_path,
        pre_sliding_crop_transform=None,
        sliding_crop=joint_transforms.SlidingCropImageOnly(713, 2 / 3.),
        input_transform=standard_transforms.ToTensor(),
        verbose=False,
        skip_if_seg_exists=False,
        use_gpu=True,
    ):
        """
        img                  - Path of input image
        save_path             - Path of output image (feature map)
        sliding_crop         - Transform that returns set of image slices
        input_transform      - Transform to apply to image before inputting to network
        skip_if_seg_exists   - Whether to overwrite or skip if segmentation exists already
        """

        if save_path is not None:
            if os.path.exists(save_path):
                if skip_if_seg_exists:
                    if verbose:
                        print(
                            "Segmentation already exists, skipping: {}".format(save_path))
                    return
                else:
                    if verbose:
                        print(
                            "Segmentation already exists, overwriting: {}".format(save_path))

        if isinstance(img_path, str):
            try:
                img = Image.open(img_path).convert('RGB')
            except OSError:
                print("Error reading input image, skipping: {}".format(img_path))
        else:
            img = img_path

        # creating sliding crop windows and transform them
        img_size_orig = img.size
        if pre_sliding_crop_transform is not None:  # might reshape image
            img = pre_sliding_crop_transform(img)

        img_slices, slices_info = sliding_crop(img)
        img_slices = [input_transform(e) for e in img_slices]
        img_slices = torch.stack(img_slices, 0)
        slices_info = torch.LongTensor(slices_info)
        slices_info.squeeze_(0)

        of_pre, oa_pre = self.net.output_features, self.net.output_all
        self.net.output_features, self.net.output_all = True, False
        feature_map = self.run_on_slices(
            img_slices, slices_info, sliding_transform_step=sliding_crop.stride_rate, use_gpu=use_gpu)
        # restore previous settings
        self.net.output_features, self.net.output_all = of_pre, oa_pre

        if save_path is not None:
            check_mkdir(os.path.dirname(save_path))
            ext = save_path.split('.')[-1]
            if ext == 'mat':
                matdict = {"features": np.transpose(feature_map, [1, 2, 0]),
                           "original_image_size": (img_size_orig[1], img_size_orig[0])}
                sio.savemat(save_path, matdict, appendmat=False)
            elif ext == 'npy':
                np.save(save_path, feature_map)
            else:
                raise ValueError(
                    'invalid file extension for save_path, only mat and np supported')

        return feature_map
