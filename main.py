# -*- coding: utf-8 -*-
from __future__ import print_function, division

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import math

from utils import tools
from datasets import  Image_from_folder
from losses import   G_loss, D_loss, L1_loss,MSE_Loss, L2
from config import config
import csv
from model import ResGenerator
from tensorboardX import SummaryWriter
import scipy.ndimage.interpolation as sni

gpuargs = config['gpuargs'] if config['cuda'] else {}

# train_dataset = MpiSintel_Test(config['train_config'])
train_dataset = Image_from_folder(config['image_folder_train'])
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True ,**gpuargs, drop_last= True)

val_dataset = Image_from_folder(config['image_folder_val'])
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False ,**gpuargs, drop_last= True)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

train_logger = SummaryWriter(log_dir = os.path.join(config['save'], 'train'), comment = 'training')
val_logger = SummaryWriter(log_dir = os.path.join(config['save'], 'val'), comment = 'validation')

#post-process is used at valitation
softmax_op = torch.nn.Softmax()
gamut = np.load('/home/chuchienshu/Documents/propagation_classification/models/custom_layers/pts_in_hull.npy')


def train_convnet(g_model,  epoch, g_criterion,  g_optimizer, dataloader ,global_iteration,  is_validate = False):
    since = time.time()
    g_total_loss = 0.
    # Each epoch has a training and validation phase
    if not is_validate:
        g_model.cuda()
        g_model.train()  # Set g_model to training mode
    else:
        g_model.cuda()
        g_model.eval()  # Set g_model to evaluate mode

    correct, total = 0, 0
    img_dir = config['save'] +'img/train/'+ str(epoch)+ '/'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Iterate over data.
    for batch_idx, ( gt, frame_1 ) in enumerate(dataloader):
        '''
        out = torchvision.utils.make_grid(torch.cat([ gt, frame_1], dim = 0),nrow= 8, pad_value=1, padding=6)
        tools.img_show(out)         
        exit() 
        '''
        # style = list(style)
        # wrap them in Variable
        if config['cuda']:
            # g_model.cuda()
            gt = Variable(gt.cuda())
            frame_1 = Variable(frame_1.cuda())
        else:
            gt = Variable(gt) 
            frame_1 = Variable(frame_1) 
        # forward
        wei_output, enc_gt = g_model(gt, frame_1)
        # backward + optimize only if in training phase
        if not is_validate:
            
            # g_model.zero_grad()
            g_optimizer.zero_grad()

            # fake_uv = g_model(real_uv)
            g_loss = g_criterion(wei_output ,enc_gt)
            g_total_loss += g_loss.data[0]

            g_loss.backward()
            g_optimizer.step()
        
        # if batch_idx %30 == 0:
        #     predic_imgs = torch.cat([img[:,:1,:,:], fake_uv], dim = 1)
        #     tools.save_imgs(predic_imgs, img_dir + str(batch_idx)+'_predic_')
        #     tools.save_imgs(img, img_dir + str(batch_idx)+'_input_')

        global_iteration += 1
        if global_iteration % config['log_frequency'] == 0:
            g_lr = g_optimizer.param_groups[0]['lr']
            # train_logger.add_image('img', img.data, global_iteration)
            # train_logger.add_image('gt', gt.data, global_iteration)
            train_logger.add_scalar('g_lr',  g_lr, global_iteration)
            train_logger.add_scalar('g_total_loss',  g_total_loss / float(batch_idx + 1), global_iteration)
        
        preface = 'training ' if not is_validate else 'validating '
        print('%s epoch %d,g_loss is %.4f' % (preface, epoch, g_total_loss/ float(batch_idx + 1)))

    time_elapsed = time.time() - since
    print('%s complete in %.0fm %.0fs'% (preface,
        time_elapsed // 60, time_elapsed % 60))
    return  global_iteration


def validation(g_model,  epoch, val_loss, dataloader ):
    since = time.time()
    total_loss = 0.
    # Each epoch has a training and validation phase
    g_model.cuda()
    g_model.eval()  # Set g_model to evaluate mode

    img_dir = config['save'] +'img/'+ str(epoch)+ '/'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    _all = len(dataloader )
    # Iterate over data.
    for batch_idx, ( gt, frame_1 ) in enumerate(dataloader):

        '''
        out = torchvision.utils.make_grid(img,nrow= 4, pad_value=1, padding=6)
        tools.img_show(out)            
        exit()      
        '''
        # style = list(style)
        # wrap them in Variable
        if config['cuda']:
            # g_model.cuda()
            gt = Variable(gt.cuda(), volatile=True)
            frame_1 = Variable(frame_1.cuda(), volatile=True)
        else:
            gt = Variable(gt) 
            frame_1 = Variable(frame_1) 
        
        # forward
        gt_img_l = gt[:,:1,:,:]
        _, _, H_orig, W_orig = gt_img_l.data.shape

        full_rs_output = g_model(gt, frame_1)
        

        # post-process
        # wei_output *= 2.606
        # wei_output = softmax_op(wei_output).cpu().data.numpy()
        full_rs_output *= 2.606
        full_rs_output = softmax_op(full_rs_output).cpu().data.numpy()

        fac_a = gamut[:,0][np.newaxis,:,np.newaxis,np.newaxis]
        fac_b = gamut[:,1][np.newaxis,:,np.newaxis,np.newaxis]

        img_l = gt_img_l.cpu().data.numpy().transpose(0,2,3,1)
        # pred_ab = np.concatenate((np.sum(wei_output * fac_a, axis=1, keepdims=True), np.sum(wei_output * fac_b, axis=1, keepdims=True)), axis=1).transpose(0,2,3,1)
        frs_pred_ab = np.concatenate((np.sum(full_rs_output * fac_a, axis=1, keepdims=True), np.sum(full_rs_output * fac_b, axis=1, keepdims=True)), axis=1).transpose(0,2,3,1)

        # _, H_out, W_out, _ = pred_ab.shape

        # # out = np.dot(wei_output, gamut)
        # ab_dec_us = sni.zoom(pred_ab,(1,1.* H_orig/H_out,1.* W_orig/W_out,1)) # upsample to match size of original image L
        '''
        v_loss = val_loss(Variable(torch.from_numpy(ab_dec_us.transpose(0,3,1,2)).cuda()), gt[:,1:,:,:])
        total_loss += v_loss.data[0]
        '''
        frs_predic_imgs = np.concatenate((img_l, frs_pred_ab ), axis = 3)
        # print(predic_imgs.shape)
        # print(batch_idx)
        tools.save_imgs(frs_predic_imgs, img_dir + str(batch_idx)+'_frspredic_')
        gt = gt.cpu().data.numpy().transpose(0,2,3,1).astype('float64')
        #lab2rgb 必须要是　ｆｌｏａｔ64 ,float32　会报错　 Images of type float must be between -1 and 1.
        tools.save_imgs(gt, img_dir + str(batch_idx)+'_gt_')

    # val_logger.add_scalar('total_loss',  total_loss / float(_all + 1), epoch)

    time_elapsed = time.time() - since
    print('%s complete in %.0fm %.0fs'% ('validation',
        time_elapsed // 60, time_elapsed % 60))
    # return  total_loss / float(_all + 1)
    return

g_model = ResGenerator()

g_optimizer = optim.Adam(g_model.parameters(), weight_decay=1e-2, betas=(0.8, 0.9))

g_scheduler = lr_scheduler.ExponentialLR(g_optimizer, gamma= 0.9)

g_criterion = G_loss()
val_loss = L2()

best_err = 1000
is_best = False
global_iteration = 0

for epoch in range(config['total_epochs']):
    print('Epoch {}/{}'.format(epoch, config['total_epochs'] - 1))
    print('-' * 10)
    g_scheduler.step()
    
    if not config['skip_training']:
        global_iteration = train_convnet(g_model,epoch, g_criterion,  g_optimizer, train_loader, global_iteration)
    
    if not config['skip_validate'] and ((epoch + 1 ) % config['validate_frequency']) == 0:
        # _error = validation(g_model,epoch, val_loss, val_loader )
        validation(g_model,epoch, val_loss, val_loader )
        # if _error < best_err:
        #     is_best = True
        # if is_best:
        _error = 0.0
        tools.save_checkpoint({   'arch' : config['model'],
                                    'epoch': epoch,
                                    'state_dict': g_model.state_dict(),
                                    'best_err': _error}, 
                                    is_best, config['save'], config['model'])
