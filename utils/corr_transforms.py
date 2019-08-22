import math
import numbers
import random

from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms.functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2, pts1, pts2, weights):
        for t in self.transforms:
            img1, img2, pts1, pts2, weights = t(img1, img2, pts1, pts2, weights)
        return img1, img2, pts1, pts2, weights

class CorrResizeOneIm(object):
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
    def __call__(self, img, pts):
        #img1, pts1
        im_orig_size = img.size
        img = F.resize(img, self.size, Image.BILINEAR)  
        im_new_size = img.size

        if im_new_size != im_orig_size:
            x_scale = im_new_size[0]/im_orig_size[0]
            y_scale = im_new_size[1]/im_orig_size[1]

            pts[0,:] = x_scale*pts[0,:]
            pts[1,:] = y_scale*pts[1,:]

        return img, pts

class CorrResize(object):
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
    def __call__(self, img1, img2, pts1, pts2, weights):
        #img1, pts1
        im1_orig_size = img1.size
        img1 = F.resize(img1, self.size, Image.BILINEAR)  
        im1_new_size = img1.size

        if im1_new_size != im1_orig_size:
            x_scale = im1_new_size[0]/im1_orig_size[0]
            y_scale = im1_new_size[1]/im1_orig_size[1]

            pts1[0,:] = x_scale*pts1[0,:]
            pts1[1,:] = y_scale*pts1[1,:]

        #img2, pts2
        im2_orig_size = img2.size
        img2 = F.resize(img2, self.size, Image.BILINEAR)  
        im2_new_size = img2.size

        if im2_new_size != im2_orig_size:
            x_scale = im2_new_size[0]/im2_orig_size[0]
            y_scale = im2_new_size[1]/im2_orig_size[1]

            pts2[0,:] = x_scale*pts2[0,:]
            pts2[1,:] = y_scale*pts2[1,:]        
         
        return img1, img2, pts1, pts2, weights         

class CorrRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1, img2, pts1, pts2, weights):
        #randomize center point
        random_ind = random.randrange(pts1.shape[1])

        mid_x1 = np.round(pts1[0, random_ind]).astype(np.int32)
        mid_y1 = np.round(pts1[1, random_ind]).astype(np.int32)
        mid_x2 = np.round(pts2[0, random_ind]).astype(np.int32)
        mid_y2 = np.round(pts2[1, random_ind]).astype(np.int32)

        if img1.size[0] < self.size[0] or img1.size[1] < self.size[1]: #needs padding
            delta_x = max(self.size[0] - img1.size[0], 0)
            delta_y = max(self.size[1] - img1.size[1], 0)
            padding = (delta_x//2, delta_y//2, delta_x-(delta_x//2), delta_y-(delta_y//2))
            img1 = ImageOps.expand(img1, padding)
            pts1[0,:] = pts1[0,:] + delta_x//2
            pts1[1,:] = pts1[1,:] + delta_y//2

        if img2.size[0] < self.size[0] or img2.size[1] < self.size[1]: #needs padding
            delta_x = max(self.size[0] - img2.size[0], 0)
            delta_y = max(self.size[1] - img2.size[1], 0)
            padding = (delta_x//2, delta_y//2, delta_x-(delta_x//2), delta_y-(delta_y//2))
            img2 = ImageOps.expand(img2, padding)
            pts2[0,:] = pts2[0,:] + delta_x//2
            pts2[1,:] = pts2[1,:] + delta_y//2            

        img1_array = np.array(img1)
        img2_array = np.array(img2)

        half_inp_size_x = np.ceil(self.size[1]/2.0).astype(np.int32)
        half_inp_size_y = np.ceil(self.size[0]/2.0).astype(np.int32)

        #crop image 1, x
        if mid_x1 - half_inp_size_x < 0:
            x_min1 = 0
            x_max1 = self.size[1]           
        elif mid_x1 + half_inp_size_x > img1_array.shape[1]:
            x_min1 = img1_array.shape[1]-self.size[1]
            x_max1 = img1_array.shape[1]
            pts1[0,:] = pts1[0,:] - (img1_array.shape[1]-self.size[1])        
        else:
            x_min1 = mid_x1 - half_inp_size_x
            x_max1 = mid_x1 - half_inp_size_x + self.size[1]
            pts1[0,:] = pts1[0,:] - (mid_x1 - half_inp_size_x)

        #crop image 1, y
        if mid_y1 - half_inp_size_y < 0:
            y_min1 = 0
            y_max1 = self.size[0]
        elif mid_y1 + half_inp_size_y > img1_array.shape[0]:
            y_min1 = img1_array.shape[0]-self.size[0]
            y_max1 = img1_array.shape[0]
            pts1[1,:] = pts1[1,:] - (img1_array.shape[0]-self.size[0])
        else:
            y_min1 = mid_y1 - half_inp_size_y
            y_max1 = mid_y1 - half_inp_size_y + self.size[0]
            pts1[1,:] = pts1[1,:] - (mid_y1 - half_inp_size_y)          

        img1_out_array = img1_array[y_min1:y_max1,x_min1:x_max1,:]                       

        #crop image 2, x
        if mid_x2 - half_inp_size_x < 0:
            x_min2 = 0
            x_max2 = self.size[1]
        elif mid_x2 + half_inp_size_x > img2_array.shape[1]:
            x_min2 = img2_array.shape[1]-self.size[1]
            x_max2 = img2_array.shape[1]
            pts2[0,:] = pts2[0,:] - (img2_array.shape[1]-self.size[1])
        else:
            x_min2 = mid_x2 - half_inp_size_x
            x_max2 = mid_x2 - half_inp_size_x + self.size[1]
            pts2[0,:] = pts2[0,:] - (mid_x2 - half_inp_size_x)

        #crop image 2, y
        if mid_y2 - half_inp_size_y < 0:
            y_min2 = 0
            y_max2 = self.size[0]
        elif mid_y2 + half_inp_size_y > img2_array.shape[0]:
            y_min2 = img2_array.shape[0]-self.size[0]
            y_max2 = img2_array.shape[0]
            pts2[1,:] = pts2[1,:] - (img2_array.shape[0]-self.size[0])
        else:
            y_min2 = mid_y2 - half_inp_size_y
            y_max2 = mid_y2 - half_inp_size_y + self.size[0]
            pts2[1,:] = pts2[1,:] - (mid_y2 - half_inp_size_y)       
      
        img2_out_array = img2_array[y_min2:y_max2,x_min2:x_max2,:] 

        # remove all correspondence not seen in cropped image
        in_im = (pts1[0,:] >= 0) & (pts1[0,:]+.5 < self.size[1]) & (pts2[0,:] >= 0) & (pts2[0,:]+.5 < self.size[1]) & (pts1[1,:] >= 0) & (pts1[1,:]+.5 < self.size[0]) & (pts2[1,:] >= 0) & (pts2[1,:]+.5 < self.size[0])
        pts1 = pts1[:,in_im]
        pts2 = pts2[:,in_im]
        weights = weights[in_im]

        img1 = Image.fromarray(img1_out_array.astype(np.uint8)).convert('RGB')            
        img2 = Image.fromarray(img2_out_array.astype(np.uint8)).convert('RGB') 

        return img1, img2, pts1, pts2, weights
