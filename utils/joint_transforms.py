import math
import numbers
import random

from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms.functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)

class Resize(object):
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
    def __call__(self, img, mask):
        return F.resize(img, self.size, Image.BILINEAR), F.resize(mask, self.size, Image.NEAREST)

class ResizeImOnly(object):
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
    def __call__(self, img, mask):
        return F.resize(img, self.size, Image.BILINEAR), mask

class RandomCropDiffSize(object):
    def __init__(self, size, scale):
        if isinstance(size, numbers.Number):
            self.size_im = (int(size), int(size))
        else:
            self.size_im = size
        
        self.scale = scale
        self.size_mask = (round(scale*self.size_im[0]), round(scale*self.size_im[1]))

    def __call__(self, img, mask):        
        w, h = img.size
        th, tw = self.size_im
        thm, twm = self.size_mask

        if w == tw and h == th:
            return img, mask
        
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        x1m = round(x1*self.scale)
        y1m = round(y1*self.scale)

        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1m, y1m, x1m + twm, y1m + thm))

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)

class RandomPerspective(object):
    def __init__(self, scale=0.25):
        self.scale = scale

    def __call__(self, img, mask):
        width, height = img.size
        a = self.scale*(random.random()-0.5) + 1
        b = self.scale*(random.random()-0.5)
        c = self.scale*(random.random()-0.5)
        d = self.scale*(random.random()-0.5)
        e = self.scale*(random.random()-0.5) + 1
        f = self.scale*(random.random()-0.5)
        g = self.scale/width*(random.random()-0.5)
        h = self.scale/height*(random.random()-0.5)
        coeffs = np.array([a, b, c, d, e, f, g, h])
        return img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC), mask.transform((width, height), Image.PERSPECTIVE, coeffs, Image.NEAREST)

class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(*self.scale(img, mask))            


class SlidingCropOld(object):
    def __init__(self, crop_size, stride_rate, ignore_label):
        self.crop_size = crop_size
        self.stride_rate = stride_rate
        self.ignore_label = ignore_label

    def _pad(self, img, mask):
        h, w = img.shape[: 2]
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant', constant_values=self.ignore_label)
        return img, mask

    def __call__(self, img, mask):
        assert img.size == mask.size

        w, h = img.size
        long_size = max(h, w)

        img = np.array(img)
        mask = np.array(mask)

        if long_size > self.crop_size:
            stride = int(math.ceil(self.crop_size * self.stride_rate))
            h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1
            w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1
            img_sublist, mask_sublist = [], []
            for yy in range(h_step_num):
                for xx in range(w_step_num):
                    sy, sx = yy * stride, xx * stride
                    ey, ex = sy + self.crop_size, sx + self.crop_size
                    img_sub = img[sy: ey, sx: ex, :]
                    mask_sub = mask[sy: ey, sx: ex]
                    img_sub, mask_sub = self._pad(img_sub, mask_sub)
                    img_sublist.append(Image.fromarray(img_sub.astype(np.uint8)).convert('RGB'))
                    mask_sublist.append(Image.fromarray(mask_sub.astype(np.uint8)).convert('P'))
            return img_sublist, mask_sublist
        else:
            img, mask = self._pad(img, mask)
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
            return img, mask


class SlidingCrop(object):
    def __init__(self, crop_size, stride_rate, ignore_label):
        self.crop_size = crop_size
        self.stride_rate = stride_rate
        self.ignore_label = ignore_label

    def _pad(self, img, mask):
        h, w = img.shape[: 2]
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant', constant_values=self.ignore_label)
        return img, mask, h, w

    def __call__(self, img, mask):
        assert img.size == mask.size

        w, h = img.size
        long_size = max(h, w)

        img = np.array(img)
        mask = np.array(mask)

        if long_size > self.crop_size:
            stride_tmp = int(math.ceil(self.crop_size * self.stride_rate))
            if h > self.crop_size:
                h_step_num = int(math.ceil((h - self.crop_size) / float(stride_tmp))) + 1
                assert h_step_num > 1
                stride_yy = (h-self.crop_size)/(h_step_num-1.)
            else:
                h_step_num = 1
                stride_yy = stride_tmp
            if w > self.crop_size:
                w_step_num = int(math.ceil((w - self.crop_size) / float(stride_tmp))) + 1
                assert w_step_num > 1
                stride_xx = (w-self.crop_size)/(w_step_num-1.)
            else:
                w_step_num = 1
                stride_xx = stride_tmp

            # Check that the actual stride is not longer than specified
            assert stride_xx <= math.ceil(self.crop_size * self.stride_rate)
            assert stride_yy <= math.ceil(self.crop_size * self.stride_rate)

            img_slices, mask_slices, slices_info = [], [], []
            for yy in range(h_step_num):
                if h < self.crop_size:
                    sy = 0
                   #ey = sy + self.crop_size # Will go beyond image
                    ey = sy + h #will not go beyond image
                elif yy < h_step_num-1:
                    sy = int(yy * stride_yy)
                    ey = sy + self.crop_size
                else:
                    # Make sure final slice does not extend beyond image
                    sy = max(h - self.crop_size, 0)
                    ey = h
                for xx in range(w_step_num):
                    if w < self.crop_size:
                        sx = 0
                        #ex = sx + self.crop_size # Will go beyond image
                        ex = sx + w #will not go beyond image
                    elif xx < w_step_num-1:
                        sx = int(xx * stride_xx)
                        ex = sx + self.crop_size
                    else:
                        # Make sure final slice does not extend beyond image
                        sx = max(w - self.crop_size, 0)
                        ex = w
                    img_sub = img[sy: ey, sx: ex, :]
                    mask_sub = mask[sy: ey, sx: ex]
                    img_sub, mask_sub, sub_h, sub_w = self._pad(img_sub, mask_sub)
                    img_slices.append(Image.fromarray(img_sub.astype(np.uint8)).convert('RGB'))
                    mask_slices.append(Image.fromarray(mask_sub.astype(np.uint8)).convert('P'))
                    slices_info.append([sy, ey, sx, ex, sub_h, sub_w])
            # print(stride_xx)
            # print(stride_yy)
            # print('\n'.join(map(str,slices_info)))
            return img_slices, mask_slices, slices_info
        else:
            img, mask, sub_h, sub_w = self._pad(img, mask)
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
            return [img], [mask], [[0, sub_h, 0, sub_w, sub_h, sub_w]]

    # def __call__(self, img, mask):
    #     assert img.size == mask.size
    # 
    #     w, h = img.size
    #     long_size = max(h, w)
    # 
    #     img = np.array(img)
    #     mask = np.array(mask)
    # 
    #     if long_size > self.crop_size:
    #         stride = int(math.ceil(self.crop_size * self.stride_rate))
    #         h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1 if h > self.crop_size else 1
    #         w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1 if w > self.crop_size else 1
    #         img_slices, mask_slices, slices_info = [], [], []
    #         for yy in range(h_step_num):
    #             for xx in range(w_step_num):
    #                 sy, sx = yy * stride, xx * stride
    #                 ey, ex = sy + self.crop_size, sx + self.crop_size
    #                 img_sub = img[sy: ey, sx: ex, :]
    #                 mask_sub = mask[sy: ey, sx: ex]
    #                 img_sub, mask_sub, sub_h, sub_w = self._pad(img_sub, mask_sub)
    #                 img_slices.append(Image.fromarray(img_sub.astype(np.uint8)).convert('RGB'))
    #                 mask_slices.append(Image.fromarray(mask_sub.astype(np.uint8)).convert('P'))
    #                 slices_info.append([sy, ey, sx, ex, sub_h, sub_w])
    #         return img_slices, mask_slices, slices_info
    #     else:
    #         img, mask, sub_h, sub_w = self._pad(img, mask)
    #         img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    #         mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    #         return [img], [mask], [[0, sub_h, 0, sub_w, sub_h, sub_w]]

class SlidingCropImageOnly(object):
    def __init__(self, crop_size, stride_rate): #stride_reate supports different strides in w and h, [stride_h, stride_w]
        self.crop_size = crop_size
        if not isinstance(stride_rate, list):
            self.stride_rate = [stride_rate, stride_rate]
        else:
            self.stride_rate = stride_rate

    def _pad(self, img):
        h, w = img.shape[: 2]
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        return img, h, w


    def __call__(self, img):
        w, h = img.size
        long_size = max(h, w)

        img = np.array(img)

        if long_size > self.crop_size:
            stride_h = int(math.ceil(self.crop_size * self.stride_rate[0]))
            stride_w = int(math.ceil(self.crop_size * self.stride_rate[1]))
            if h > self.crop_size:
                h_step_num = int(math.ceil((h - self.crop_size) / float(stride_h))) + 1
                assert h_step_num > 1
                stride_yy = (h-self.crop_size)/(h_step_num-1.)
            else:
                h_step_num = 1
                stride_yy = stride_tmp
            if w > self.crop_size:
                w_step_num = int(math.ceil((w - self.crop_size) / float(stride_w))) + 1
                assert w_step_num > 1
                stride_xx = (w-self.crop_size)/(w_step_num-1.)
            else:
                w_step_num = 1
                stride_xx = stride_tmp

            # Check that the actual stride is not longer than specified
            assert stride_xx <= math.ceil(self.crop_size * self.stride_rate[0])
            assert stride_yy <= math.ceil(self.crop_size * self.stride_rate[1])

            img_slices, mask_slices, slices_info = [], [], []
            for yy in range(h_step_num):
                if h < self.crop_size:
                    sy = 0
                   #ey = sy + self.crop_size # Will go beyond image
                    ey = sy + h #will not go beyond image
                elif yy < h_step_num-1:
                    sy = int(yy * stride_yy)
                    ey = sy + self.crop_size
                else:
                    # Make sure final slice does not extend beyond image
                    sy = max(h - self.crop_size, 0)
                    ey = h
                for xx in range(w_step_num):
                    if w < self.crop_size:
                        sx = 0
                        #ex = sx + self.crop_size # Will go beyond image
                        ex = sx + w #will not go beyond image
                    elif xx < w_step_num-1:
                        sx = int(xx * stride_xx)
                        ex = sx + self.crop_size
                    else:
                        # Make sure final slice does not extend beyond image
                        sx = max(w - self.crop_size, 0)
                        ex = w
                    img_sub = img[sy: ey, sx: ex, :]
                    img_sub, sub_h, sub_w = self._pad(img_sub)
                    img_slices.append(Image.fromarray(img_sub.astype(np.uint8)).convert('RGB'))
                    slices_info.append([sy, ey, sx, ex, sub_h, sub_w])
            return img_slices, slices_info
        else:
            img, mask, sub_h, sub_w = self._pad(img)
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            return [img], [[0, sub_h, 0, sub_w, sub_h, sub_w]]                               
