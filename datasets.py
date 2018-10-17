# from __future__ import division
import torch
import torch.utils.data as data
from skimage.color import rgb2gray, rgb2yuv, yuv2rgb, rgb2lab, lab2rgb
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
# import cv2

import os, math, random
from os.path import *
import numpy as np

from glob import glob
# import utils.frame_utils as frame_utils
from utils import frame_utils
from scipy.misc import imread, imresize
from utils.tools import resize_flow, warp_forward_flow
import math
from utils.img_transforms import *
import natsort


class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[int((self.h-self.th)//2):int((self.h+self.th)//2), int((self.w-self.tw)//2):int((self.w+self.tw)//2),:]

comp = Compose([
    Scale([286, 572]),
    RandomCrop([256,256]),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    # RandomRotate(11)
])


class Image_from_folder(data.Dataset):

    def __init__(self, args ):
        super().__init__()
        self.render_size = args['render_size']
        self.replicates = args['replicates']
        self.frame_1 = []
        self.gt_images = []

        o_f = open(args['dstype'], 'r')

        lines = o_f.readlines()
        for l in lines:
            frame_, gt_img = l.split()
            self.frame_1.append(frame_)
            self.gt_images.append(gt_img)
            # print( img, gt_img ) 

        # _len = (len(self.frame_1) // 10) * 9
        # _len = len(self.frame_1) - 8
        
        self.train = args['train']
        # if self.train:
        #     self.gt_images = self.gt_images[:_len]
        #     self.frame_1 = self.frame_1[:_len]
        # else:
        #     self.gt_images = self.gt_images[_len:]
        #     self.frame_1 = self.frame_1[_len:]
        '''
        for i, g, k in zip(self.images, self.gt_images, self.frame_1):
            print(i, g, k)
        exit()
        '''
        assert len(self.frame_1) == len(self.gt_images)
        # print(len(self.images))
        # exit()
        self.size = len(self.frame_1)
        self.frame_size = frame_utils.read_gen(self.frame_1[0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%32) or (self.frame_size[1]%32):

            self.render_size[0] = ((self.frame_size[0])//64)  *64
            self.render_size[1] = ( (self.frame_size[1])//64)  * 64

    def __getitem__(self, index):

        index = index % self.size

        img2 = frame_utils.read_gen(self.gt_images[index])
        img3 = frame_utils.read_gen(self.frame_1[index])

        img2 = resize(img2, self.render_size)   
        img3 = resize(img3, self.render_size)   

        #more data agumentation
        if self.train:
            img2, img3 = comp([img2, img3])
            # pass
        # else:
        #     img2 = resize(img2, [256,256])
        #     img3 = resize(img3, [256,256])
            # img1, img2 = comp([img1, img2])
        # else:
        #     img1 = resize(img1, [256,256])     
        #     img2 = resize(img2, [256,256])
        img2 = rgb2lab(img2)    
        img3 = rgb2lab(img3)   
        # print(img2, img2.dtype)
        # print(lab2rgb(img2), ' ljga')
        # exit()
        img2 = np.array(img2).transpose(2,0,1)
        img3 = np.array(img3).transpose(2,0,1)

        img2 = torch.from_numpy(img2.astype(np.float32))
        img3 = torch.from_numpy(img3.astype(np.float32))

        return  img2, img3

    def __len__(self):
        return self.size * self.replicates


