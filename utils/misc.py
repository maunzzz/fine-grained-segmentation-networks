import os
import json
from math import ceil

import re
import sys
# import psutil
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image
import colorsys

root = os.path.join(os.path.dirname(__file__), '..')


def get_root():
    return root


def get_global_opts():
    # Default options:
    opts = {
        "result_path": 'dummy/path',
        "cityscapes_path": 'dummy/path',
        "vistas_path": 'dummy/path',
        "wildash_root_path": 'dummy/path',
        "robotcar_root_path": 'dummy/path',
        "robotcar_corr_path": 'dummy/path',
        "robotcar_im_path": 'dummy/path',
        "cmu_root_path": 'dummy/path',
        "cmu_corr_path": 'dummy/path',
        "cmu_im_path": 'dummy/path',
    }

    global_opts_path = os.path.join(get_root(), 'global_opts.json')
    if os.path.exists(global_opts_path):
        with open(global_opts_path, 'r') as opts_file:
            json_opts = json.load(opts_file)
        opts.update(json_opts)
    return opts


def rename_key_of_ordered_dict(ordered_dict, old_name, new_name):
    return OrderedDict([(new_name, v) if k == old_name else (k, v) for k, v in ordered_dict.items()])


def rename_keys_to_match(state_dict):
    state_dict = rename_key_of_ordered_dict(state_dict, 'final.conv6.bias', 'conv6.bias')
    state_dict = rename_key_of_ordered_dict(state_dict, 'final.conv6.weight', 'conv6.weight')
    state_dict = rename_key_of_ordered_dict(state_dict, 'aux.conv6_1.bias', 'conv6_1.bias')
    state_dict = rename_key_of_ordered_dict(state_dict, 'aux.conv6_1.weight', 'conv6_1.weight')
    return state_dict


def replace_root(old_path, old_root, new_root):
    assert old_path[:len(old_root)] == old_root
    # Ensure same format (trailing slash for both or neither)
    assert old_root.endswith(os.sep) == new_root.endswith(os.sep)
    relpath = old_path[len(old_root):]
    new_path = new_root + relpath
    return new_path


def replace_suffix(old_str, old_suffix, new_suffix):
    assert old_str[-len(old_suffix):] == old_suffix
    return old_str[:-len(old_suffix)] + new_suffix


def absorb_bn(module, bn_module):
    w = module.weight.data
    if module.bias is None:
        zeros = torch.Tensor(module.out_channels).zero_().type(w.type())
        module.bias = nn.Parameter(zeros)
    b = module.bias.data
    invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
    w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w))
    b.add_(-bn_module.running_mean).mul_(invstd)

    if bn_module.affine:
        w.mul_(bn_module.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
        b.mul_(bn_module.weight.data).add_(bn_module.bias.data)

    bn_module.register_buffer('running_mean', None)
    bn_module.register_buffer('running_var', None)
    bn_module.register_parameter('weight', None)
    bn_module.register_parameter('bias', None)
    bn_module.affine = False


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)


def search_absorbe_bn(model):
    prev = None
    for m in model.children():
        if is_bn(m) and is_absorbing(prev):
            absorb_bn(prev, m)
        search_absorbe_bn(m)
        prev = m


def add_bias(net):
    for module in net.modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and module.bias is None:
            w = module.weight.data
            zeros = torch.Tensor(module.out_channels).zero_().type(w.type())
            module.bias = nn.Parameter(zeros)


def freeze_bn(net):
    """ Freezes batchnorm modules during training, useful when training with small batches"""
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm1d):
            module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm3d):
            module.eval()


def check_mkdir(dir_name):
    os.makedirs(dir_name, exist_ok=True)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


def fast_hist_cluster(label_pred, label_true, num_classes, num_clusters):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_clusters * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes * num_clusters).reshape(num_classes, num_clusters)
    return hist


def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

# predictions and gts should have size (N,W,H), where N is the number of images in a batch
# lists_of_classes: list that contains subsets of classes that will have individual means taken (stationary/non-stationary)


def evaluate_incremental(hist, predictions, gts, num_classes, lists_of_classes=None):
    # hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    present_classes = hist.sum(axis=1) != 0

    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)

    # acc_cls = np.nanmean(acc_cls)
    acc_cls_mean = np.mean(acc_cls[present_classes])

    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    # mean_iu = np.nanmean(iu)
    mean_iu = np.mean(iu[present_classes])

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    if lists_of_classes is None:
        div = None
    else:
        div = []
        for class_list in lists_of_classes:
            tmp_present_classes = np.zeros_like(present_classes)
            tmp_present_classes[class_list] = present_classes[class_list]
            div.append({'miou': np.mean(iu[tmp_present_classes]), 'm_acc_cls': np.mean(acc_cls[present_classes])})

    return acc, acc_cls_mean, mean_iu, fwavacc, hist, div


def evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    present_classes = hist.sum(axis=1) != 0

    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    # acc_cls = np.nanmean(acc_cls)
    acc_cls = np.mean(acc_cls[present_classes])
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    # mean_iu = np.nanmean(iu)
    mean_iu = np.mean(iu[present_classes])

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, hist


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PolyLR(object):
    def __init__(self, optimizer, curr_iter, max_iter, lr_decay):
        self.max_iter = float(max_iter)
        self.init_lr_groups = []
        for p in optimizer.param_groups:
            self.init_lr_groups.append(p['lr'])
        self.param_groups = optimizer.param_groups
        self.curr_iter = curr_iter
        self.lr_decay = lr_decay

    def step(self):
        for idx, p in enumerate(self.param_groups):
            p['lr'] = self.init_lr_groups[idx] * (1 - self.curr_iter / self.max_iter) ** self.lr_decay


def get_network_name_from_iteration(folder_path, iteration):
    for f in os.listdir(folder_path):
        rem = re.match(r'^iter_(\d+)_.*.pth$', f)
        if rem:
            it = int(rem.group(1))
            if it == iteration:
                return(rem.group(0))
    return None


def get_latest_network_name(folder_path):
    net_to_load = ''
    max_it = 0
    for f in os.listdir(folder_path):
        rem = re.match(r'^iter_(\d+)_.*.pth$', f)
        if rem:
            it = int(rem.group(1))
            if it > max_it:
                net_to_load = rem.group(0)
                max_it = it

    return net_to_load


def collect_gt_from_slices(gt_slices, slices_info):
    imsize1 = slices_info[0, :, 1].max().item()
    imsize2 = slices_info[0, :, 3].max().item()

    gts_tmp = np.zeros((1, imsize1, imsize2), dtype=int)
    gt.transpose_(0, 1)
    slices_info.squeeze_(0)
    count = torch.zeros(imsize1, imsize2)
    for gt_slice, info in zip(input, gt, slices_info):
        gts_tmp[0, info[0]: info[1], info[2]: info[3]] += gt_slice[0, :info[4], :info[5]].data.numpy()
        count[info[0]: info[1], info[2]: info[3]] += 1

    output /= count
    gts_tmp //= count.numpy().astype(int)
    return gts_tmp


def clean_log_before_continuing(log_path, last_val_iter):
    pat = re.compile(r"\[iter (\d+) / (\d+)\]")
    lines_to_keep = []
    with open(log_path) as f:
        for line in f:
            mm = pat.match(line)
            if mm:
                this_iter = int(mm.group(1))
                if this_iter > last_val_iter:
                    break
            lines_to_keep.append(line)
    with open(log_path, 'w') as f:
        for line in lines_to_keep:
            f.write(line)


def im_to_ext_name(im_name, ext):
    if ext[0] == '.':
        ext = ext[1:]
    im_name_split = im_name.split('/')
    return im_name_split[0] + '_' + im_name_split[2][:-3] + ext


def load_resnet101_weights(net, state_dict):
    net_dict = net.state_dict()
    for key in net_dict.keys():
        if key.startswith('fcn') and ('num_batches_tracked' not in key):
            s_key = key[4:]
            net_dict[key] = state_dict[s_key].data
            if (len(net_dict[key].size()) > 1) and (net_dict[key].size(1) == 3):  # if first image
                net_dict[key] /= 255.0  # to counteract difference in input normalization

    # this should actually be net_dict since we transferred the values from state_dict already
    net.load_state_dict(net_dict)


def remap_mask(mask, direction, ignore_label):
    # function to map mask from 0,1,2,255 to 0,1,2,3 and back again, this makes all new pixels introduced by transforms to be ignored during training
    new_vals = np.array(mask.getdata())
    if direction == 0:
        new_vals = new_vals + 1
        new_vals[new_vals == ignore_label+1] = 0
    else:
        new_vals[new_vals == 0] = ignore_label+1
        new_vals = new_vals - 1
    s1, s2 = mask.size
    new_vals = np.reshape(new_vals, (s2, s1))
    return Image.fromarray(np.uint8(new_vals), 'L')


def log_and_print(str2write, f_handle):
    print(str2write)
    f_handle.write(str2write + "\n")


def colorize(imn, n_classes, saturation=0.6, value=0.65):
    if n_classes <= 256:
        palette = create_palette(n_classes, saturation, value)
        imn = imn.convert('P')
        imn.putpalette(palette)
    else:
        imn = add_color_to_image(imn, n_classes, saturation, value)

    return imn


def create_palette(n_classes, saturation=0.6, value=0.65):
    palette = [0]*256*3
    HSV_tuples = [(x*1.0/n_classes, saturation, value) for x in range(n_classes)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    for i in range(n_classes):
        palette[3*i + 0] = RGB_tuples[i][0]
        palette[3*i + 1] = RGB_tuples[i][1]
        palette[3*i + 2] = RGB_tuples[i][2]
    return [int(256*cc) for cc in palette]


def add_color_to_image(im_in, n_classes, saturation=0.6, value=0.65):
    arr = np.array(im_in)
    rgb_array = np.zeros(arr.shape + (3,))
    HSV_tuples = [(x*1.0/n_classes, saturation, value) for x in range(n_classes)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            rgb_array[y, x, 0] = RGB_tuples[arr[y, x]][0]
            rgb_array[y, x, 1] = RGB_tuples[arr[y, x]][1]
            rgb_array[y, x, 2] = RGB_tuples[arr[y, x]][2]

    return Image.fromarray((rgb_array*256).astype(np.uint8), mode="RGB")
