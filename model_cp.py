from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

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

def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD

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

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
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
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

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

class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.conv1_1 = nn.Conv2d(4, 64, 3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
        
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
        self.pool1 = nn.AvgPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=2, dilation=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
        self.pool2 = nn.AvgPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=2, dilation=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=2, dilation=2)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
        self.pool3 = nn.AvgPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=2, dilation=2)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=2 , dilation=2)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
        self.pool4 = nn.AvgPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.bn9 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        # 
        self.relu5_2 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
        self.pool5 = nn.AvgPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # delete vgg's fc layer, adding customize layers #*****************
        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=5, stride=2, padding=2, dilation=1, output_padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.relu6 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)

        self.deconv2 = nn.ConvTranspose2d(
            1024, 256, kernel_size=5, stride=2, padding=2, dilation=1, output_padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.relu7 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
        self.deconv3 = nn.ConvTranspose2d(
            512, 128, kernel_size=5, stride=2, padding=2, dilation=1, output_padding=1)
        self.bn12 = nn.BatchNorm2d(128)
        self.relu8 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)

        self.deconv4 = nn.ConvTranspose2d(
            256, 64, kernel_size=5, stride=2, padding=2, dilation=1, output_padding=1)
        self.bn13 = nn.BatchNorm2d(64)
        self.relu9 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
        self.deconv5 = nn.ConvTranspose2d(
            128, 32, kernel_size=5, stride=2, padding=2, dilation=1, output_padding=1)
        self.bn14 = nn.BatchNorm2d(32)
        self.relu10 = nn.LeakyReLU(negative_slope= 0.2, inplace= True)

        self.generator = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

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
    
    def forward(self, input):
        rgb_mean = input.contiguous().view(input.size()[:2]+(-1,)).mean(dim=-1).view(input.size()[:2] + (1,1,))
        
        b = self.corr(input,input)
        print(b.data.shape)
        x = (input - rgb_mean) 
        # conv1_1 = self.relu1_1(self.conv1_1(input[:,:1, :, :]))
        conv1_1 = self.relu1_1(self.bn1( self.conv1_1(x)))
        conv1_2 = self.pool1( self.relu1_2(self.bn2( self.conv1_2( conv1_1 ))) )
        # print('conv1_2 ', conv1_2.data.shape)
        # conv1_2  (8L, 64L, 128L, 128L)

        conv2_1 = self.relu2_1(self.bn3( self.conv2_1( conv1_2 )))
        conv2_2 = self.pool2(self.relu2_2(self.bn4( self.conv2_2( conv2_1 ))))
        # conv2_2  torch.Size([8, 128, 104, 256])

        conv3_1 = self.relu3_1(self.bn5( self.conv3_1(conv2_2)))
        conv3_2 = self.pool3(self.relu3_2(self.bn6( self.conv3_2(conv3_1))))
        # conv3_2  torch.Size([8, 256, 104, 256])

        conv4_1 = self.relu4_1(self.bn7( self.conv4_1(conv3_2)))
        # print('conv4_1 ', conv4_1.data.shape)
        conv4_2 = self.pool4(self.relu4_2(self.bn8( self.conv4_2(conv4_1))))
        # conv4_2  torch.Size([8, 512, 52, 128])

        conv5_1 = self.relu5_1(self.bn9( self.conv5_1(conv4_2)))
        conv5_2 = self.pool5(self.relu5_2(self.bn10( self.conv5_2(conv5_1))))
        # conv5_2  torch.Size([8, 512, 26, 64])

        deconv1 = self.relu6(self.deconv1(conv5_2))

        cat1 = torch.cat([deconv1, conv4_2], dim = 1)
        deconv2 = self.relu7(self.deconv2(cat1))

        cat2 = torch.cat([deconv2, conv3_2],dim = 1)
        deconv3 = self.relu8(self.deconv3(cat2))

        cat3 = torch.cat([deconv3, conv2_2],dim = 1)
        deconv4 = self.relu9(self.deconv4(cat3))

        cat4 = torch.cat([deconv4, conv1_2], dim = 1)
        deconv5 = self.relu10(self.deconv5(cat4))

        # gen = torch.cat([input[:, :1, :,:], self.generator(deconv5)],dim = 1) 
        gen = self.generator(deconv5)
        return self.tanh(gen)
    '''

    def forward(self, input):
        # conv1_1 = self.relu1_1(self.conv1_1(input[:,:1, :, :]))
        conv1_1 = self.relu1_1(self.bn1( self.conv1_1(input)))
        conv1_2 = self.pool1( self.relu1_2(self.bn2( self.conv1_2( conv1_1 ))) )

        conv2_1 = self.pool2(self.relu2_1(self.bn3( self.conv2_1( conv1_2 ))))

        conv3_1 = self.pool3(self.relu3_1(self.bn5( self.conv3_1(conv2_1))))

        conv4_1 = self.pool4(self.relu4_1(self.bn7( self.conv4_1(conv3_1))))

        conv5_1 = self.pool5(self.relu5_1(self.bn9( self.conv5_1(conv4_1))))

        deconv1 = self.relu6(self.bn10( self.deconv1(conv5_1)))

        cat1 = torch.cat([deconv1, conv4_1], dim = 1)
        deconv2 = self.relu7(self.bn11( self.deconv2(cat1)))

        cat2 = torch.cat([deconv2, conv3_1],dim = 1)
        deconv3 = self.relu8(self.bn12( self.deconv3(cat2)))

        cat3 = torch.cat([deconv3, conv2_1],dim = 1)
        deconv4 = self.relu9(self.bn13( self.deconv4(cat3)))

        cat4 = torch.cat([deconv4, conv1_2], dim = 1)
        deconv5 = self.relu10(self.bn14( self.deconv5(cat4)))
        gen = self.generator(deconv5)
        return self.tanh(gen)
    '''
def _make_layer(in_planes, out_planes, kernel_size=3, stride=1, bias = True):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
            nn.LeakyReLU(0.2,inplace=True)
        )

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lay1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2,inplace=True)
        )

        self.lay2 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, stride=2,   padding=1),
            nn.LeakyReLU(0.2,inplace=True)
        )

        self.lay3 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=3, stride=2,   padding=1),
            nn.LeakyReLU(0.2,inplace=True)
        )

        self.lay4 = nn.Sequential(
            nn.Conv2d(256,512, kernel_size=3, stride=2,  padding=1),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.pool = nn.AvgPool2d([8,8], stride= 8)

        self.fc = nn.Linear(2048, 1)
        self.tanh = nn.Sigmoid()

    def forward(self, input):

        lay1 = self.lay1(input)
        lay2 = self.lay2(lay1)
        lay3 = self.lay3(lay2)
        lay4 = self.pool( self.lay4(lay3))
         
        b, _, _, _ = lay4.data.shape
        
        fc = self.fc(lay4.view([b, -1]))
        return self.tanh(fc)