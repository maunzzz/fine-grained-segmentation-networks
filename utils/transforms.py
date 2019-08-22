import random

import numpy as np
from skimage.filters import gaussian
import torch
from PIL import Image, ImageFilter


class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class FreeScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = tuple(reversed(size))  # size: (h, w)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR)

class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))


class RandomGaussianBlur(object):
    def __call__(self, img):
        sigma = 0.15 + random.random() * 1.15
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))

class RGB2BGR(object):
    """Changes color channel mod from RGB to BGR on a CxHxW image"""
    def __call__(self, img):   
        assert img.size(0) == 3 
        idx = [i for i in range(img.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        inverted_tensor = img.index_select(0, idx)
        return inverted_tensor