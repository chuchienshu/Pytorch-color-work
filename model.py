from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools

import math
import numpy as np

import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
from config import config

import matplotlib.pyplot as plt
from scipy.misc import imread
from skimage.color import rgb2gray, rgb2yuv, yuv2rgb
from models.correlation_package.modules.correlation import Correlation
from models.custom_layers.trainable_layers import *
from torch.autograd import Variable

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        
        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)]
        model += [nn.ReLU()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResGenerator(nn.Module):

    def __init__(self, padding=1, dilation=1):
        super().__init__()
        self.nnecnclayer = NNEncLayer()
        self.priorboostlayer = PriorBoostLayer()
        self.nongraymasklayer = NonGrayMaskLayer()
        # self.rebalancelayer = ClassRebalanceMultLayer()
        self.rebalancelayer = Rebalance_Op.apply
        self.relu = nn.ReLU(  inplace= True)

        self.rg = ResnetGenerator(96,313, n_blocks=13, use_dropout=True)

        # Rebalance_Op.apply
        self.pool = nn.AvgPool2d(4,4)
        self.pool1 = nn.AvgPool2d(2,2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        
        #------------------- process frame 1
        self.conv1_1_ = nn.Conv2d(3, 64, 3, padding=padding, dilation=dilation)
        self.bn1_ = nn.BatchNorm2d(64)
        
        self.conv1_2_ = nn.Conv2d(64, 64, 3,stride=2, padding=1, dilation=1)
        self.bn2_ = nn.BatchNorm2d(64)

        # conv2
        self.conv2_1_ = nn.Conv2d(64, 128, 3, padding=padding, dilation=dilation)
        self.bn3_ = nn.BatchNorm2d(128)
        self.conv2_2_ = nn.Conv2d(128, 128, 3,stride=2, padding=1, dilation=1)
        self.bn4_ = nn.BatchNorm2d(128)
        #------------------- process frame 1
        self.added_conv1 = nn.Conv2d(64, 64, 1)
        self.added_bn1 = nn.BatchNorm2d(64)

        self.added_conv2 = nn.Conv2d(441, 32, 1)
        self.added_bn2 = nn.BatchNorm2d(32)
        
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv1_2 = nn.Conv2d(64, 64, 3,stride=2, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(64)


        '''
        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=padding, dilation=dilation)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(  inplace= True)
        self.conv2_2 = nn.Conv2d(128, 128, 3,stride=2, padding=1, dilation=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(  inplace= True)

        self.generator = nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()
        '''
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, gt_img, frame_1):
        # rgb_mean = input.contiguous().view(input.size()[:2]+(-1,)).mean(dim=-1).view(input.size()[:2] + (1,1,))
        
        # b = self.corr(input,input)
        # x = (input - rgb_mean) 
        
        #make gt_img_l between in [-50, 50] and 0-center.
        gt_img_l = (gt_img[:,:1,:,:] - 50.)
        # frame1_ab = frame_1[:,1:,:,:]

        # ------------------conv frame1_ab
        conv1_1_ = self.relu(self.bn1_( self.conv1_1_( frame_1)))
        conv1_2_ = self.relu(self.bn2_( self.conv1_2_( conv1_1_ )))

        # ------------------conv frame1_ab

        # conv1_1 = self.relu1_1(self.conv1_1(input[:,:1, :, :]))
        conv1_1 = self.relu(self.bn1( self.conv1_1(gt_img_l)))
        conv1_2 = self.relu(self.bn2( self.conv1_2( conv1_1 )))
        # print('conv1_2 ', conv1_2.data.shape)
        
        corr = self.corr(conv1_2_, conv1_2)#从前者里面抽取 n*c*1*1 的filter 与后者做卷积
        conv1_2c = self.relu(self.added_bn1( self.added_conv1(conv1_2)))
        corr = self.relu(self.added_bn2( self.added_conv2(corr)))
        cat_corr = torch.cat([corr, conv1_2c], dim = 1)
        
        # print(cat_corr.data.shape)

        gen = self.rg(cat_corr)
        # print(out.data.shape)

        # gen = self.generator(deconv5)
        # print(gen.data.shape, ' gen')

        if self.training:

            # ********************** process gtimg_ab *************
            gt_img_ab = self.pool1(gt_img[:,1:,:,:]).cpu().data.numpy()
            
            # self.nnecnclayer  
            # self.priorboostlayer  
            # self.nongraymasklayer
            enc = self.nnecnclayer(gt_img_ab)
            # (4, 313, 256, 256)
            ngm = self.nongraymasklayer(gt_img_ab)
            # (4, 1, 1, 1)  ngm
            pb = self.priorboostlayer(enc)
            #(4, 1, 256, 256)  pb
            boost_factor = (pb * ngm).astype('float32')
            # (4, 1, 256, 256)
            # *******************************************
            # print(boost_factor.dtype)
            boost_factor = Variable(torch.from_numpy(boost_factor).cuda(), requires_grad = False)

            wei_output = self.rebalancelayer(gen, boost_factor)
            #(4, 313, 64, 64)  enc

            return wei_output, Variable(torch.from_numpy(enc).cuda())
        else:
            return self.upsample(gen)


'''
# for test
input_ = Variable(torch.ones(2, 3, 256,256))
rg = ResnetGenerator(3,6)
mods = rg.modules()
for m in mods:
    print(m)
out = rg(input_)
print(out.data.shape)
'''