'''
Portions of this code copyright 2017, Clement Pinard
'''

# freda (todo) : adversarial loss 

import torch
import torch.nn as nn
import math

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L1_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, fake, real):

        return self.loss(fake, real)

class MSE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, fake, real):

        return self.loss(fake, real)


class D_loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()
        # nn.BCEWithLogitsLoss()
        
    def forward(self, real_disc, fake_disc):

        true_loss = self.loss(real_disc, torch.ones_like(real_disc))
        fake_loss = self.loss(fake_disc, torch.zeros_like(fake_disc))

        return true_loss*1.2 + fake_loss

class G_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.sf = nn.Softmax2d()
        # nn.loss
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, input, target):
        n, c, h, w = target.data.shape

        input = input.permute(0,2,3,1).contiguous().view(n*h*w, -1)
        target = target.permute(0,2,3,1).contiguous().view(n*h*w, -1)
        #[262144, 313]
        
        return self.loss(input, torch.max(target, 1)[1])

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    

    