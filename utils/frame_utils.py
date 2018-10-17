# -*- coding: utf-8 -*-
import numpy as np
from os.path import *
from scipy.misc import imread
# import flow_utils #在Python 2里面这样跨文件导入没问题，但是3里面就不行，3中要看做一个整体的文件

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imread(file_name)
        if im.shape[2] > 3:
            return im[:,:,:3]
        else:
            return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    return []
