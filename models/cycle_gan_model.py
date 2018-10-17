import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
from PIL import Image


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A_1 = networks.define_G_narrow(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        
        self.netG_A_2 = networks.define_G(opt.input_nc, opt.output_nc,
                                              opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)        
        
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A_1 = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_A_2 = networks.define_D(opt.output_nc, opt.ndf,
                                                    opt.which_model_netD,
                                                    opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)            
            
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A_1, 'G_A_1', which_epoch)
            self.load_network(self.netG_A_2, 'G_A_2', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A_1, 'D_A_1', which_epoch)
                self.load_network(self.netD_A_2, 'D_A_2', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_narrow_32_pool = ImagePool(opt.pool_size)
            self.fake_B_narrow_44_pool = ImagePool(opt.pool_size)
            self.fake_B_narrow_64_pool = ImagePool(opt.pool_size)
            self.fake_B_32_processed_pool = ImagePool(opt.pool_size)
            self.fake_B_44_processed_pool = ImagePool(opt.pool_size)
            self.fake_B_64_processed_pool = ImagePool(opt.pool_size)
            
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G_A_1 = torch.optim.Adam(self.netG_A_1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_A_2 = torch.optim.Adam(self.netG_A_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_B = torch.optim.Adam(self.netG_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))            
            self.optimizer_D_A_1 = torch.optim.Adam(self.netD_A_1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A_2 = torch.optim.Adam(self.netD_A_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G_A_1)
            self.optimizers.append(self.optimizer_G_A_2)
            self.optimizers.append(self.optimizer_D_A_1)
            self.optimizers.append(self.optimizer_D_A_2)            
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A_1)
        networks.print_network(self.netG_A_2)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A_1)
            networks.print_network(self.netD_A_2)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        
        input_A_32 = input['A' if AtoB else 'B'][0].numpy()
        input_A_32_model = np.zeros((input_A_32.shape[0], 32, 32))
        for c in range(input_A_32.shape[0]):
            input_A_32_model[c, :, :] = np.asarray(Image.fromarray(input_A_32[c, :, :]).resize((32, 32), Image.NEAREST))            
        input_A_32 = torch.from_numpy(input_A_32_model)
        input_A_32 = input_A_32.view(-1, 3, 32, 32) 

        input_A_44 = input['A' if AtoB else 'B'][0].numpy()
        input_A_44_model = np.zeros((input_A_44.shape[0], 44, 44))
        for c in range(input_A_44.shape[0]):
            input_A_44_model[c, :, :] = np.asarray(Image.fromarray(input_A_44[c, :, :]).resize((44, 44), Image.NEAREST))            
        input_A_44 = torch.from_numpy(input_A_44_model)
        input_A_44 = input_A_44.view(-1, 3, 44, 44)         
        
        input_A_64 = input['A' if AtoB else 'B'][0].numpy()
        input_A_64_model = np.zeros((input_A_64.shape[0], 64, 64))
        for c in range(input_A_64.shape[0]):
            input_A_64_model[c, :, :] = np.asarray(Image.fromarray(input_A_64[c, :, :]).resize((64, 64), Image.NEAREST))            
        input_A_64 = torch.from_numpy(input_A_64_model)
        input_A_64 = input_A_64.view(-1, 3, 64, 64)                
        
        input_B = input['B' if AtoB else 'A']
        
        input_B_32 = input['B' if AtoB else 'A'][0].numpy()
        input_B_32_model = np.zeros((input_B_32.shape[0], 32, 32))
        for c in range(input_B_32.shape[0]):
            input_B_32_model[c, :, :] = np.asarray(Image.fromarray(input_B_32[c, :, :]).resize((32, 32), Image.NEAREST))            
        input_B_32 = torch.from_numpy(input_B_32_model)
        input_B_32 = input_B_32.view(-1, 3, 32, 32)
        
        input_B_44 = input['B' if AtoB else 'A'][0].numpy()
        input_B_44_model = np.zeros((input_B_44.shape[0], 44, 44))
        for c in range(input_B_44.shape[0]):
            input_B_44_model[c, :, :] = np.asarray(Image.fromarray(input_B_44[c, :, :]).resize((44, 44), Image.NEAREST))            
        input_B_44 = torch.from_numpy(input_B_44_model)
        input_B_44 = input_B_44.view(-1, 3, 44, 44)          
        
        input_B_64 = input['B' if AtoB else 'A'][0].numpy()
        input_B_64_model = np.zeros((input_B_64.shape[0], 64, 64))
        for c in range(input_B_64.shape[0]):
            input_B_64_model[c, :, :] = np.asarray(Image.fromarray(input_B_64[c, :, :]).resize((64, 64), Image.NEAREST))            
        input_B_64 = torch.from_numpy(input_B_64_model)
        input_B_64 = input_B_64.view(-1, 3, 64, 64)        
        
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_A_32 = input_A_32.cuda(self.gpu_ids[0], async=True)
            input_A_44 = input_A_44.cuda(self.gpu_ids[0], async=True) 
            input_A_64 = input_A_64.cuda(self.gpu_ids[0], async=True)                                         
            
            
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
            input_B_32 = input_B_32.cuda(self.gpu_ids[0], async=True)
            input_B_44 = input_B_44.cuda(self.gpu_ids[0], async=True)   
            input_B_64 = input_B_64.cuda(self.gpu_ids[0], async=True)           
            
        self.input_A = input_A
        self.input_A_32 = input_A_32
        self.input_A_44 = input_A_44
        self.input_A_64 = input_A_64
        self.input_B = input_B
        self.input_B_32 = input_B_32
        self.input_B_44 = input_B_44
        self.input_B_64 = input_B_64
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_A_32 = Variable(self.input_A_32)
        self.real_A_44 = Variable(self.input_A_44) 
        self.real_A_64 = Variable(self.input_A_64)
        
        self.real_B = Variable(self.input_B)
        self.real_B_32 = Variable(self.input_B_32)
        self.real_B_44 = Variable(self.input_B_44)
        self.real_B_64 = Variable(self.input_B_64)

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        fake_B_narrow_32, fake_B_narrow_44, fake_B_narrow_64 = self.netG_A_1(real_A)
        fake_B_32 = np.zeros((fake_B_narrow_32.data[0].shape[0], 256, 256))
        for c in range(fake_B_32.shape[0]):
            fake_B_32[c, :, :] = np.asarray(Image.fromarray(fake_B_narrow_32.data[0].cpu().numpy()[c, :, :]).resize((256, 256), Image.NEAREST))            
        fake_B_32 = torch.from_numpy(fake_B_32).float()
        fake_B_32 = Variable(fake_B_32.view(-1, 3, 256, 256).cuda())
        
        fake_B_44 = np.zeros((fake_B_narrow_44.data[0].shape[0], 256, 256))
        for c in range(fake_B_44.shape[0]):
            fake_B_44[c, :, :] = np.asarray(Image.fromarray(fake_B_narrow_44.data[0].cpu().numpy()[c, :, :]).resize((256, 256), Image.NEAREST))            
        fake_B_44 = torch.from_numpy(fake_B_44).float()
        fake_B_44 = Variable(fake_B_44.view(-1, 3, 256, 256).cuda())        
        
        fake_B_64 = np.zeros((fake_B_narrow_64.data[0].shape[0], 256, 256))
        for c in range(fake_B_64.shape[0]):
            fake_B_64[c, :, :] = np.asarray(Image.fromarray(fake_B_narrow_64.data[0].cpu().numpy()[c, :, :]).resize((256, 256), Image.NEAREST))            
        fake_B_64 = torch.from_numpy(fake_B_64).float()
        fake_B_64 = Variable(fake_B_64.view(-1, 3, 256, 256).cuda()) 
        
        fake_B_32_processed = self.netG_A_2(fake_B_32)
        fake_B_44_processed = self.netG_A_2(fake_B_44)
        fake_B_64_processed = self.netG_A_2(fake_B_64)
        rec_A = self.netG_B(fake_B_64_processed)
        self.fake_B_32 = fake_B_32.data
        self.fake_B_44 = fake_B_44.data
        self.fake_B_64 = fake_B_64.data
        self.fake_B_32_processed = fake_B_32_processed.data
        self.fake_B_44_processed = fake_B_44_processed.data
        self.fake_B_64_processed = fake_B_64_processed.data
        self.rec_A = rec_A.data
        
        real_B = Variable(self.input_B, volatile=True)
        fake_A = self.netG_B(real_B)
        rec_B_narrow_32, rec_B_narrow_44, rec_B_narrow_64 = self.netG_A_1(fake_A)
        rec_B_64 = np.zeros((rec_B_narrow_64.data[0].shape[0], 256, 256))
        for c in range(rec_B_64.shape[0]):
            rec_B_64[c, :, :] = np.asarray(Image.fromarray(rec_B_narrow_64.data[0].cpu().numpy()[c, :, :]).resize((256, 256), Image.NEAREST))            
        rec_B_64 = torch.from_numpy(rec_B_64).float()
        rec_B_64 = Variable(rec_B_64.view(-1, 3, 256, 256).cuda())
        rec_B_64_processed = self.netG_A_2(rec_B_64)
        self.fake_A = fake_A.data
        self.rec_B_narrow_64 = rec_B_narrow_64.data
        self.rec_B_64 = rec_B_64.data
        self.rec_B_64_processed = rec_B_64_processed.data
        

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A_1(self):
        fake_B_narrow_32 = self.fake_B_narrow_32_pool.query(self.fake_B_narrow_32)
        fake_B_narrow_44 = self.fake_B_narrow_44_pool.query(self.fake_B_narrow_44)
        fake_B_narrow_64 = self.fake_B_narrow_64_pool.query(self.fake_B_narrow_64)        
        loss_D_A_1 = self.backward_D_basic(self.netD_A_1, self.real_B_32.float(), fake_B_narrow_32)
        loss_D_A_1 += self.backward_D_basic(self.netD_A_1, self.real_B_44.float(), fake_B_narrow_44)
        loss_D_A_1 += self.backward_D_basic(self.netD_A_1, self.real_B_64.float(), fake_B_narrow_64)
        
        self.loss_D_A_1 = loss_D_A_1.data[0]
        
    def backward_D_A_2(self):
        fake_B_32_processed = self.fake_B_32_processed_pool.query(self.fake_B_32_processed)
        fake_B_44_processed = self.fake_B_44_processed_pool.query(self.fake_B_44_processed)
        fake_B_64_processed = self.fake_B_64_processed_pool.query(self.fake_B_64_processed)
        loss_D_A_2 = self.backward_D_basic(self.netD_A_2, self.real_B.float(), fake_B_32_processed)
        loss_D_A_2 += self.backward_D_basic(self.netD_A_2, self.real_B.float(), fake_B_44_processed)
        loss_D_A_2 += self.backward_D_basic(self.netD_A_2, self.real_B.float(), fake_B_64_processed)
        self.loss_D_A_2 = loss_D_A_2.data[0]

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.data[0]

    def backward_G(self):
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B        
        '''
        lambda_idt = self.opt.lambda_identity
        
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG_A(self.real_B)
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        '''

        # GAN loss D_A_1(G_A_1(A))
        fake_B_narrow_32, fake_B_narrow_44, fake_B_narrow_64 = self.netG_A_1(self.real_A)
        pred_fake_32 = self.netD_A_1(fake_B_narrow_32)
        pred_fake_44 = self.netD_A_1(fake_B_narrow_44)
        pred_fake_64 = self.netD_A_1(fake_B_narrow_64)        
        
        loss_G_A_1 = self.criterionGAN(pred_fake_32, True)
        loss_G_A_1 += self.criterionGAN(pred_fake_44, True)
        loss_G_A_1 += self.criterionGAN(pred_fake_64, True)
        
        # GAN loss D_A_2(G_A_2(A))
        fake_B_32 = np.zeros((fake_B_narrow_32.data[0].shape[0], 256, 256))
        for c in range(fake_B_32.shape[0]):
            fake_B_32[c, :, :] = np.asarray(Image.fromarray(fake_B_narrow_32.data[0].cpu().numpy()[c, :, :]).resize((256, 256), Image.NEAREST))            
        fake_B_32 = torch.from_numpy(fake_B_32).float()
        fake_B_32 = Variable(fake_B_32.view(-1, 3, 256, 256).cuda())
        
        fake_B_32_processed = self.netG_A_2(fake_B_32)
        pred_fake_32 = self.netD_A_2(fake_B_32_processed)
        
        fake_B_44 = np.zeros((fake_B_narrow_44.data[0].shape[0], 256, 256))
        for c in range(fake_B_44.shape[0]):
            fake_B_44[c, :, :] = np.asarray(Image.fromarray(fake_B_narrow_44.data[0].cpu().numpy()[c, :, :]).resize((256, 256), Image.NEAREST))            
        fake_B_44 = torch.from_numpy(fake_B_44).float()
        fake_B_44 = Variable(fake_B_44.view(-1, 3, 256, 256).cuda())
        
        fake_B_44_processed = self.netG_A_2(fake_B_44)        
        pred_fake_44 = self.netD_A_2(fake_B_44_processed)        
        
        fake_B_64 = np.zeros((fake_B_narrow_64.data[0].shape[0], 256, 256))
        for c in range(fake_B_64.shape[0]):
            fake_B_64[c, :, :] = np.asarray(Image.fromarray(fake_B_narrow_64.data[0].cpu().numpy()[c, :, :]).resize((256, 256), Image.NEAREST))            
        fake_B_64 = torch.from_numpy(fake_B_64).float()
        fake_B_64 = Variable(fake_B_64.view(-1, 3, 256, 256).cuda())
        
        fake_B_64_processed = self.netG_A_2(fake_B_64)
        pred_fake_64 = self.netD_A_2(fake_B_64_processed)        
        
        loss_G_A_2 = self.criterionGAN(pred_fake_32, True)
        loss_G_A_2 += self.criterionGAN(pred_fake_44, True)
        loss_G_A_2 += self.criterionGAN(pred_fake_64, True)

        # GAN loss D_B(G_B(B))
        fake_A = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        rec_A = self.netG_B(fake_B_64_processed)
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A)

        # Backward cycle loss
        rec_B_narrow_32, rec_B_narrow_44, rec_B_narrow_64 = self.netG_A_1(fake_A)
        loss_cycle_B_1 = self.criterionCycle(rec_B_narrow_32, self.real_B_32.float())
        loss_cycle_B_1 += self.criterionCycle(rec_B_narrow_44, self.real_B_44.float())
        loss_cycle_B_1 += self.criterionCycle(rec_B_narrow_64, self.real_B_64.float())
        
        rec_B_64 = np.zeros((rec_B_narrow_64.data[0].shape[0], 256, 256))
        for c in range(rec_B_64.shape[0]):
            rec_B_64[c, :, :] = np.asarray(Image.fromarray(rec_B_narrow_64.data[0].cpu().numpy()[c, :, :]).resize((256, 256), Image.NEAREST))            
        rec_B_64 = torch.from_numpy(rec_B_64).float()
        rec_B_64 = Variable(rec_B_64.view(-1, 3, 256, 256).cuda())
        rec_B_64_processed = self.netG_A_2(rec_B_64)
        loss_cycle_B_2 = self.criterionCycle(rec_B_64_processed, self.real_B)
        # combined loss
        loss_G_A_1_IO_L1 = self.criterionL1(fake_B_narrow_32, self.real_A_32.float())
        loss_G_A_1_IO_L1 += self.criterionL1(fake_B_narrow_44, self.real_A_44.float())
        loss_G_A_1_IO_L1 += self.criterionL1(fake_B_narrow_64, self.real_A_64.float())
        self.loss_G_A_1_ = loss_G_A_1 \
                         + loss_cycle_B_1 * lambda_A \
                         + loss_G_A_1_IO_L1 * lambda_A
        
        loss_G_A_2_IO_L1 = self.criterionL1(fake_B_32_processed, fake_B_32)
        loss_G_A_2_IO_L1 += self.criterionL1(fake_B_44_processed, fake_B_44)
        loss_G_A_2_IO_L1 += self.criterionL1(fake_B_64_processed, fake_B_64)
        self.loss_G_A_2_ = loss_G_A_2 \
                         + loss_cycle_B_2 * lambda_A \
                         + loss_G_A_2_IO_L1
        
        loss_G_B_IO_L1 = self.criterionL1(fake_A, self.real_B)
        self.loss_G_B_ = loss_G_B \
                       + loss_cycle_A * lambda_B \
                       + loss_G_B_IO_L1 * lambda_B

        self.fake_B_narrow_32 = fake_B_narrow_32.data
        self.fake_B_narrow_44 = fake_B_narrow_44.data
        self.fake_B_narrow_64 = fake_B_narrow_64.data
        self.fake_B_32 = fake_B_32.data
        self.fake_B_44 = fake_B_44.data
        self.fake_B_64 = fake_B_64.data
        self.fake_B_32_processed = fake_B_32_processed.data
        self.fake_B_44_processed = fake_B_44_processed.data
        self.fake_B_64_processed = fake_B_64_processed.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B_64 = rec_B_64.data
        self.rec_B_64_processed = rec_B_64_processed.data

        self.loss_G_A_1 = loss_G_A_1.data[0]
        self.loss_G_A_2 = loss_G_A_2.data[0]
        self.loss_G_B = loss_G_B.data[0]
        
        self.loss_cycle_B_1 = loss_cycle_B_1.data[0] * lambda_A
        self.loss_G_A_1_IO_L1 = loss_G_A_1_IO_L1.data[0] * lambda_A
        self.loss_cycle_B_2 = loss_cycle_B_2.data[0] * lambda_A
        self.loss_G_A_2_IO_L1 = loss_G_A_2_IO_L1.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0] * lambda_B
        self.loss_G_B_IO_L1 = loss_G_B_IO_L1.data[0] * lambda_B

    def optimize_parameters(self):
        # forward
        self.forward()
        # backward
        self.backward_G()        
        # G_A_1
        self.optimizer_G_A_1.zero_grad()
        self.loss_G_A_1_.backward(retain_graph=True)
        self.optimizer_G_A_1.step()
        # G_A_2
        self.optimizer_G_A_2.zero_grad()
        self.loss_G_A_2_.backward(retain_graph=True)
        self.optimizer_G_A_2.step()
        # G_B
        self.optimizer_G_B.zero_grad()
        self.loss_G_B_.backward()
        self.optimizer_G_B.step()
        # D_A_1
        self.optimizer_D_A_1.zero_grad()
        self.backward_D_A_1()
        self.optimizer_D_A_1.step()
        # D_A_2
        self.optimizer_D_A_2.zero_grad()
        self.backward_D_A_2()
        self.optimizer_D_A_2.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A_1', self.loss_D_A_1), ('G_A_1', self.loss_G_A_1), 
                                  ('D_A_2', self.loss_D_A_2),('G_A_2', self.loss_G_A_2), 
                                  ('D_B', self.loss_D_B), ('G_B', self.loss_G_B),
                                  ('cycle_B_1', self.loss_cycle_B_1), ('G_A_1_IO_L1', self.loss_G_A_1_IO_L1),
                                  ('cycle_B_2', self.loss_cycle_B_2), ('G_A_2_IO_L1', self.loss_G_A_2_IO_L1),
                                  ('cycle_A', self.loss_cycle_A), ('G_B_IO_L1', self.loss_G_B_IO_L1)])
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B_32 = util.tensor2im(self.fake_B_32)
        fake_B_44 = util.tensor2im(self.fake_B_44)
        fake_B_64 = util.tensor2im(self.fake_B_64)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B_64 = util.tensor2im(self.rec_B_64)
        
        fake_B_32_processed = util.tensor2im(self.fake_B_32_processed)
        fake_B_44_processed = util.tensor2im(self.fake_B_44_processed)
        fake_B_64_processed = util.tensor2im(self.fake_B_64_processed)
        rec_B_64 = util.tensor2im(self.rec_B_64)
        rec_B_64_processed = util.tensor2im(self.rec_B_64_processed)
        
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B_32', fake_B_32), ('fake_B_44', fake_B_44), ('fake_B_64', fake_B_64), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B_64', rec_B_64), ('rec_B_64_processed', rec_B_64_processed),
                                   ('fake_B_32_processed', fake_B_32_processed), ('fake_B_44_processed', fake_B_44_processed), ('fake_B_64_processed', fake_B_64_processed)])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A_1, 'G_A_1', label, self.gpu_ids)
        self.save_network(self.netG_A_2, 'G_A_2', label, self.gpu_ids)
        self.save_network(self.netD_A_1, 'D_A_1', label, self.gpu_ids)
        self.save_network(self.netD_A_2, 'D_A_2', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        
