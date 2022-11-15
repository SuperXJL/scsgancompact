# -- coding: utf-8 --
# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import functools
import math
import time

import imageio
import numpy as np
import torch
import os
from tqdm import tqdm
from torch import nn, optim
from torch.autograd import Variable

import scipy.io as sio

from components.base_blocks import ZF_equalization, MMSE_equalization, LMMSE_channel_est, LS_channel_est, \
    cal_gradient_penalty, init_net, get_norm_layer, GANLoss, Normalize, get_scheduler, CreateMISOEnv
from components.base_models import BaseModel, Encoder, NLayerDiscriminator, Subnet, OFDM, Generator, SRGenerator, \
    Channel


def adjust_learning_rate(optimizers, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = start_lr * (0.9 ** (epoch // 100))
    lr = start_lr * (0.95 ** (epoch // 50))
    for optimizer in optimizers:
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class Solver(BaseModel):

    def __init__(self, config, voc_train_loader, voc_test_loader, voc_valid_loader):
        BaseModel.__init__(self, config)
        # 加载预先存储好的H
        try:
            self.H_record = np.load("record_H_record.npz")['H'][0, :]
            print("加载H_record成功，记录时间共%d秒，%d BS，%d UE，%d Antennas"%(self.H_record.shape[0],self.H_record.shape[1],self.H_record.shape[2],self.H_record.shape[3]))
        except:
            print("加载H_record失败")
            exit(0)

        self.voc_train_loader = voc_train_loader
        self.voc_test_loader = voc_test_loader
        self.voc_valid_loader = voc_valid_loader
        self.is_Feat = config.is_Feat
        self.compression_rate = config.compression_rate
        self.validate_epoch = config.validate_epoch
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.isTrain = config.isTrain
        self.dataset = config.dataset
        self.gan_mode = config.gan_mode
        # self.train_iters = config.train_iters
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.log_epoch = config.log_epoch
        self.sample_epoch = config.sample_epoch
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.continue_learning = config.continue_learning
        self.scale_rate = config.scale_rate
        """Builds a generator and a discriminator."""
        self.last_epoch = config.last_epoch
        self.train_acc = []
        self.train_acc_local = []
        self.Channel = Channel(config, self.device)
        self.OFDM = OFDM(config, self.Channel, 'Pilot_bit.pt')
        self.EQ = config.EQ
        self.normalize = Normalize()
        self.MSE = nn.MSELoss()
        self.criterionGAN = GANLoss(config.gan_mode).to(self.device)
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.optimizers = []
        self.n_epochs = config.n_epochs
        self.lambda_L2 = config.lambda_L2
        self.P = config.P
        self.K = config.K
        self.M = config.M
        self.S = config.S
        self.L = config.L
        self.CE = config.CE
        self.SNR = config.SNR
        self.N_pilot = config.N_pilot
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.first_kernel = config.first_kernel
        self.init_type = config.init_type
        self.init_gain = config.init_gain
        self.input_channel = config.input_channel
        self.output_channel = config.output_channel
        self.conv_out_channel = config.conv_out_channel
        self.conv_max_out_channel = config.conv_max_out_channel
        self.C_channel = config.C_channel
        # self.resnet_block_num=config.resnet_block_num
        self.n_downsample = config.n_downsample
        self.norm_EG = config.norm_EG
        self.activation = config.activation
        self.is_feedback = config.is_feedback
        self.feedforward = config.feedforward
        self.n_layers_D = config.n_layers_D
        self.H_record =self.H_record[:, 0, 0, :self.L]
          
        self.H_record = torch.tensor(data= self.H_record.reshape([self.H_record.shape[0],1, 1, 1, self.L]), dtype=torch.complex64).repeat(1,self.batch_size, 1, self.P, 1).view(self.H_record.shape[0],self.batch_size, self.P, self.L, 1)
        self.H_record = torch.cat((self.H_record, torch.zeros((self.H_record.shape[0],self.batch_size, self.P, self.M - self.L, 1))), 3).cuda()
        self.H_record = torch.cat([self.H_record.real, self.H_record.imag], -1)
        if self.is_feedback:
            add_C = self.P
            # add_C = self.P * 2
        else:
            add_C = 0
        
        # define networks (both generator and discriminator)
        # 设置信道编码器
        self.CHEncoder = init_net(
            Encoder(input_nc=self.input_channel, ngf=self.conv_out_channel, max_ngf=self.conv_max_out_channel,
                    input_channel=self.C_channel,
                    n_downsampling=self.n_downsample,
                    norm_layer=get_norm_layer(norm_type=self.norm_EG), padding_type="reflect",
                    first_kernel=self.first_kernel, first_add_C=add_C),
            self.init_type,
            self.init_gain,
            self.gpu_ids)

        self.Generator = init_net(
            Generator(output_nc=self.output_channel, ngf=self.conv_out_channel, max_ngf=self.conv_max_out_channel,
                      input_channel=2*self.C_channel,
                      n_downsampling=self.n_downsample,
                      norm_layer=get_norm_layer(norm_type=self.norm_EG), padding_type="reflect",
                      first_kernel=config.first_kernel, activation_=self.activation),
            self.init_type,
            self.init_gain,
            self.gpu_ids)

        self.SRGenerator = init_net(
            SRGenerator(output_nc=self.output_channel, ngf=self.conv_out_channel, max_ngf=self.conv_max_out_channel,
                        input_channel=3,
                        n_downsampling=self.n_downsample, norm_layer=get_norm_layer(norm_type=self.norm_EG),
                        padding_type="reflect",
                        activation_=self.activation),
            self.init_type,
            self.init_gain,
            self.gpu_ids
        )

        self.Discriminator = init_net(
            NLayerDiscriminator(self.input_channel, self.conv_out_channel, n_layers=self.n_layers_D,
                                norm_layer=get_norm_layer(norm_type=self.norm_EG)),
            self.init_type, self.init_gain, self.gpu_ids)

        if torch.cuda.is_available():
            self.CHEncoder = self.CHEncoder.cuda()
            self.Generator = self.Generator.cuda()
            self.Discriminator = self.Discriminator.cuda()

        # g_params = list(self.g12.parameters()) + list(self.g21.parameters())
        # d_params = list(self.d1.parameters()) + list(self.d2.parameters())

        if self.feedforward in ['EXPLICIT-RES']:

            norm_layer = get_norm_layer(norm_type=self.norm_EG)
            if type(norm_layer) == functools.partial:
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d

            self.SubNet1 = init_net(Subnet(dim=(self.N_pilot * self.P + 1),
                                           dim_out=self.P, padding_type='zero', norm_layer=norm_layer,
                                           use_dropout=False,
                                           use_bias=use_bias), self.init_type, self.init_gain, self.gpu_ids)
            # self.SubNet1 = init_net(Subnet(dim=(self.N_pilot * self.P + 1) * 2,
            #                              dim_out=self.P * 2, padding_type='zero', norm_layer=norm_layer, use_dropout=False,
            #                  use_bias=use_bias), self.init_type, self.init_gain, self.gpu_ids)

            self.SubNet2 = init_net(Subnet(dim=(self.S + 1) * self.P,
                                           dim_out=self.S * self.P, padding_type='zero', norm_layer=norm_layer,
                                           use_dropout=False,
                                           use_bias=use_bias), self.init_type, self.init_gain, self.gpu_ids)
            # self.SubNet2 = init_net(Subnet(dim=(self.S + 1) * self.P*2,
            #                                dim_out=self.S * self.P * 2, padding_type='zero', norm_layer=norm_layer,
            #                                use_dropout=False,
            #                                use_bias=use_bias), self.init_type, self.init_gain, self.gpu_ids)

        print('---------- Networks initialized -------------')

        # set loss functions and optimizers

        g_params = list(self.CHEncoder.parameters()) + list(self.Generator.parameters())

        if self.config.feedforward in ['EXPLICIT-RES']:
            g_params += list(self.SubNet1.parameters()) + list(self.SubNet2.parameters())
        elif self.config.feedforward in ['EXPLICIT-RES2']:
            g_params += list(self.SubNet.parameters())

        # self.optimizer_G = torch.optim.Adam(params, lr=config.lr, betas=(config.beta1, 0.999))
        e_params = list(self.CHEncoder.parameters())
        # g_params = list(self.Generator.parameters())
        d_params = list(self.Discriminator.parameters())
        self.optimizer_E = optim.Adam(e_params, self.lr, [self.beta1, self.beta2])
        self.optimizer_G = optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        self.optimizer_D = optim.Adam(d_params, self.lr, [self.beta1, self.beta2])
        self.optimizers.append(self.optimizer_E)
        self.optimizers.append(self.optimizer_G)

        if self.config.gan_mode != 'none':
            params = list(self.Discriminator.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, config) for optimizer in self.optimizers]
        if not self.isTrain or config.continue_train:
            load_suffix = 'iter_%d' % config.load_iter if config.load_iter > 0 else config.epoch
            self.load_networks(load_suffix)

    def merge_images(self, sources, targets, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([3, row * h, row * w * 2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
        return merged.transpose(1, 2, 0)

    def merge_images_encoder(self, sources, targets, decoder, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([3, row * h, row * w * 3])
        for idx, (s, t, de) in enumerate(zip(sources, targets, decoder)):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 3) * h:(j * 3 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 3 + 1) * h:(j * 3 + 2) * h] = t
            merged[:, i * h:(i + 1) * h, (j * 3 + 2) * h:(j * 3 + 3) * h] = de
        return merged.transpose(1, 2, 0)

    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

    def train(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % device)
        print('compression rate:', self.compression_rate)
        print('learning rate = %.7f' % self.lr)
        voc_iterator = iter(self.voc_train_loader)
        iter_per_epoch = len(iter(self.voc_train_loader))
        psnr_low2high_record = []

        if (self.continue_learning):
            # last_epoch= self.load_model_path.split('-')[-1]
            self.CHEncoder.load_state_dict(
                #torch.load(os.path.join(self.model_path, self.dataset + 'CHEncoder-%d.pkl' % (self.last_epoch))))
                torch.load(os.path.join(self.model_path, self.dataset + 'best_CHEncoder_%.2f.pkl'%(self.scale_rate))))
            self.Generator.load_state_dict(
                #torch.load(os.path.join(self.model_path, self.dataset + 'Generator-%d.pkl' % (self.last_epoch))))
                torch.load(os.path.join(self.model_path, self.dataset + 'best_Generator_%.2f.pkl'%(self.scale_rate) )))
            self.Discriminator.load_state_dict(
                #torch.load(os.path.join(self.model_path, self.dataset + 'Discriminator-%d.pkl' % (self.last_epoch))))
                torch.load(os.path.join(self.model_path, self.dataset + 'best_Discriminator_%.2f.pkl'%(self.scale_rate) )))
            print("成功加载模型，继续训练,当前起始step为:", self.last_epoch)
        self.psnr_record = []
        self.g_loss_record = []
        self.d_fake_loss_record = []
        self.d_real_loss_record = []

        self.best_g_loss_valid = 9999

        # for voc, l_voc in tqdm(self.voc_train_loader):
        iter_per_epoch = len(self.voc_train_loader)
        print("iter_per_epoch:%d ,total_epochs: %d" % (iter_per_epoch, self.n_epochs))
        


        for epoch in range(self.n_epochs):
            # print("current epoch = ",epoch)
            adjust_learning_rate(self.optimizers,epoch,self.lr )
 
            #print("Epoch:{}  Lr:{:.2E}".format(epoch,))

            batch_fake = []
            batch_src = []
            batch_low = []
            iter_count = 0
            sum_l_g_f = 0
            sum_l_d_f = 0
            sum_l_d_t = 0
            sum_psnr_low2high = 0
            tqdm_loader = tqdm(self.voc_train_loader)
            for voc, l_voc in tqdm_loader:
                iter_count += 1
                N = voc.shape[0]
                voc = voc.cuda()
                l_voc = l_voc.cuda()
              
                #G2 = torch.tensor(data= self.H_record[iter_count, 0, 0, :].reshape([1, 1, 1, self.L]), dtype=torch.complex64).repeat(N, 1, self.P, 1).view(N, self.P, self.L, 1)
                G2 = self.H_record[iter_count,:N,:]
                # cof = None
                
                latent = self.CHEncoder(l_voc)

                tx = latent.view(N, self.P, self.S, 2, self.M).permute(0, 1, 2, 4, 3)

                out_pilot, out_sig, noise_pwr, self.PAPR, self.PAPR_cp = self.OFDM(
                    self.normalize(tx, 1),
                    K=self.K,
                    SNR=self.SNR,
                    cof=G2)

                N, C, H, W = latent.shape

                if self.feedforward == 'IMPLICIT':
                    r1 = self.OFDM.pilot.repeat(N, self.P, 1, 1, 1)
                    r2 = out_pilot
                    r3 = out_sig
                    dec_in = torch.cat((r1, r2, r3), 2).contiguous().permute(0, 1, 2, 4, 3).contiguous().view(N, -1, H,
                                                                                                              W)
                    fake = self.Generator(dec_in)
                    # fake = self.SRGenerator(restore)
                elif self.feedforward == 'EXPLICIT-CE':
                    # Channel estimation
                    if self.CE == 'LS':
                        self.H_est = LS_channel_est(self.OFDM.pilot, out_pilot)
                    elif self.CE == 'LMMSE':
                        self.H_est = LMMSE_channel_est(self.OFDM.pilot, out_pilot, self.M * noise_pwr)
                    elif self.CE == 'TRUE':
                        self.H_est = self.H_true.unsqueeze(2).to(self.device)
                    else:
                        raise NotImplementedError('The channel estimation method [%s] is not implemented' % self.CE)
                    r1 = self.H_est
                    r2 = out_pilot
                    r3 = out_sig
                    dec_in = torch.cat((r1, r2, r3), 2).contiguous().permute(0, 1, 2, 4, 3).contiguous().view(N, -1, H,
                                                                                                              W)
                    fake = self.Generator(dec_in)
                elif self.feedforward == 'EXPLICIT-RES':
                    # Channel estimation
                    if self.CE == 'LS':
                        self.H_est = LS_channel_est(self.OFDM.pilot, out_pilot)
                    elif self.CE == 'LMMSE':
                        self.H_est = LMMSE_channel_est(self.OFDM.pilot, out_pilot, self.M * noise_pwr)
                    elif self.CE == 'TRUE':
                        self.H_est = self.H_true.unsqueeze(2).to(self.device)
                    else:
                        raise NotImplementedError('The channel estimation method [%s] is not implemented' % self.CE)

                    sub11 = self.OFDM.pilot.repeat(N, 1, 1, 1, 1)
                    sub12 = out_pilot
                    sub1_input = torch.cat((sub11, sub12), 2).contiguous().permute(0, 1, 2, 4, 3).contiguous().view(N,
                                                                                                                    -1,
                                                                                                                    H,
                                                                                                                    W)
                    sub1_output = self.SubNet1(sub1_input)
                    sub1_output = sub1_output.view(N, self.P, 1, 2, self.M).permute(0, 1, 2, 4, 3)
                    # self.equalization(self.H_est + sub1_output, out_sig, noise_pwr)

                    # Equalization
                    if self.EQ == 'ZF':
                        self.rx = ZF_equalization(self.H_est, out_sig)
                    elif self.EQ == 'MMSE':
                        self.rx = MMSE_equalization(self.H_est, out_sig, self.config.M * noise_pwr)
                    elif self.EQ == 'None':
                        self.rx = None
                    else:
                        raise NotImplementedError('The equalization method [%s] is not implemented' % self.CE)

                    sub21 = self.H_est + sub1_output
                    sub22 = out_sig
                    sub2_input = torch.cat((sub21, sub22), 2).contiguous().permute(0, 1, 2, 4, 3).contiguous().view(N,
                                                                                                                    -1,
                                                                                                                    H,
                                                                                                                    W)
                    sub2_output = self.SubNet2(sub2_input)
                    sub2_output = sub2_output.view(N, self.P, self.S, 2, self.M).permute(0, 1, 2, 4, 3)
                    dec_in = (self.rx + sub2_output).permute(0, 1, 2, 4, 3).contiguous().view(latent.shape)
                    fake = self.Generator(dec_in)
                    # fake = self.SRGenerator(restore)

                if (N == self.batch_size):
                    batch_fake = fake
                    batch_src = voc
                    batch_low = l_voc

                mse = self.MSE(fake, voc)
                if (mse > 0):
                    psnr_low2high = 10 * np.log10(
                        np.max(fake.cpu().detach().numpy()) ** 2 / mse.cpu().detach().numpy() ** 2)
                    sum_psnr_low2high += psnr_low2high
                # update D
                # self.loss_D_fake,self.loss_D_real = self.update_discriminator()
                if self.gan_mode != 'none':
                    self.optimizer_D.zero_grad()  # set D's gradients to zero
                    # calculate gradients for D
                    self.set_requires_grad(self.Discriminator, True)  # enable backprop for D
                    _, pred_fake = self.Discriminator(fake.detach())
                    self.loss_D_fake = self.criterionGAN(pred_fake, False)
                    _, pred_real = self.Discriminator(voc)
                    self.loss_D_real = self.criterionGAN(pred_real, True)

                    if self.gan_mode in ['lsgan', 'vanilla']:
                        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
                        self.loss_D.backward()
                    elif self.gan_mode == 'wgangp':
                        penalty, grad = cal_gradient_penalty(self.Discriminator, voc, fake.detach(), self.device,
                                                             type='mixed', constant=1.0, lambda_gp=10.0)
                        self.loss_D = self.loss_D_fake + self.loss_D_real + penalty
                        self.loss_D.backward(retain_graph=True)

                    self.optimizer_D.step()  # update D's weights
                    self.set_requires_grad(self.Discriminator, False)  # enable backprop for D
                # update G
                # self.set_requires_grad(self.Discriminator, False)  # enable backprop for D
                # self.set_requires_grad(self.Generator, True)  # D requires no gradients when optimizing G
                loss_G_GAN = 0
                loss_G_Feat = 0

                self.optimizer_G.zero_grad()  # set G's gradients to zero
                # calculate graidents for G
                if self.gan_mode != 'none':
                    feat_fake, pred_fake = self.Discriminator(fake)
                    self.loss_G_GAN = self.criterionGAN(pred_fake, False)

                    if self.is_Feat:
                        feat_real, pred_real = self.Discriminator(voc)
                        self.loss_G_Feat = 0

                        for j in range(len(feat_real)):
                            self.loss_G_Feat += self.criterionFeat(feat_real[j].detach(),
                                                                   feat_fake[j]) * self.config.lambda_feat
                    else:
                        self.loss_G_Feat = 0
                    self.loss_G_L2 = self.criterionL2(fake, voc) * self.lambda_L2
                    self.loss_PAPR = torch.mean(self.PAPR_cp)
                    self.loss_G = self.loss_G_GAN + self.loss_G_Feat + self.loss_G_L2
                    self.loss_G.backward()
                    self.optimizer_G.step()  # udpate G's weights

                sum_l_g_f += self.loss_G.item()
                sum_l_d_f += self.loss_D_fake.item()
                sum_l_d_t += self.loss_D_real.item()
                info = (
                        'TRAIN PHASE Epoch [%d/%d], lr: %.8f, d_real_loss: %.6f, ' 'd_fake_loss: %.6f, g_loss: %.4f, psnr_low2high: %.4f'
                        % (
                            epoch + 1, self.n_epochs + self.last_epoch,self.optimizers[0].state_dict()['param_groups'][0]['lr'], sum_l_d_t / iter_count, sum_l_d_f / iter_count,
                            sum_l_g_f / iter_count, sum_psnr_low2high / iter_count))
                tqdm_loader.set_description_str(info)

            self.psnr_record.append(sum_psnr_low2high / iter_count)
            self.g_loss_record.append(sum_l_g_f / iter_count)
            self.d_fake_loss_record.append(sum_l_d_f / iter_count)
            self.d_real_loss_record.append(sum_l_d_t / iter_count)

            # print the log info
            if (epoch + 1) % self.validate_epoch == 0:
                tqdm_loader_val = tqdm(self.voc_valid_loader)
                self.validate(epoch, tqdm_loader_val)

            if (epoch + 1) % self.sample_epoch == 0:
                np.savez(os.path.join(self.model_path, self.dataset + 'records_ep%d_%.2f' % (epoch + 1,self.scale_rate)),
                         psnr=np.array(self.psnr_record), g_loss=np.array(self.g_loss_record),
                         d_fake_loss=np.array(self.d_fake_loss_record), d_real_loss=np.array(self.d_real_loss_record))

                # save images
                # fake = fake.cuda()
                # out_encoder_reshape_data = self.to_data(out_encoder_reshape)

                # low and fixed-high
                merged = self.merge_images(batch_low.cpu().detach().numpy(),
                                           batch_fake.cpu().detach().numpy())
                path = os.path.join(self.sample_path,
                                    'sample' + self.dataset + '-%d-lh-%.2f.png' % (epoch + 1 + self.last_epoch,self.scale_rate))
                # scipy.misc.imsave(path, merged)
                imageio.imwrite(path, merged)
                print('saved %s' % path)

                # high and fixed-high
                merged = self.merge_images(batch_src.cpu().detach().numpy(), batch_fake.cpu().detach().numpy())
                path = os.path.join(self.sample_path,
                                    'sample' + self.dataset + '-%d-hh-%.2f.png' % (epoch + 1 + self.last_epoch,self.scale_rate))
                # scipy.misc.imsave(path, merged)
                imageio.imwrite(path, merged)
                print('saved %s' % path)
                # save saved_models
                # save the model parameters for each epoch
                CHEncoder_path = os.path.join(self.model_path,
                                              self.dataset + 'CHEncoder-%d-%.2f.pkl' % (epoch + 1 + self.last_epoch,self.scale_rate))
                Generator_path = os.path.join(self.model_path,
                                              self.dataset + 'Generator-%d-%.2f.pkl' % (epoch + 1 + self.last_epoch,self.scale_rate))
                Discriminator_path = os.path.join(self.model_path,
                                                  self.dataset + 'Discriminator-%d-%.2f.pkl' % (epoch + 1 + self.last_epoch,self.scale_rate))
                torch.save(self.CHEncoder.state_dict(), CHEncoder_path)
                torch.save(self.Generator.state_dict(), Generator_path)
                torch.save(self.Discriminator.state_dict(), Discriminator_path)
            #        info =  ('Epoch [%d/%d], d_real_loss: %.6f, ' 'd_fake_loss: %.6f, g_loss: %.4f, mean_psnr_low2high: %.4f'
            #              % ( epoch + 1, self.n_epochs + self.last_epoch, self.loss_D_real.item(), self.loss_D_fake.item(),
            #              self.loss_G_L2.item(), np.array(psnr_low2high_record[epoch - self.last_epoch]).mean()))
            #        tqdm_loader.set_description_str(info)
            # save the sampled images

            # if epoch % self.print_freq == 0:  # print training losses and save logging information to the disk
            #             losses = self.get_current_losses()
            #             print("step: ", epoch, " loss: ", losses)

        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

    def validate(self, epoch, tqdm_loader):
        sum_l_g_f = 0
        sum_l_d_f = 0
        sum_l_d_t = 0
        iter_count = 0
        sum_psnr_low2high = 0
   
        for voc, l_voc in tqdm_loader:
            iter_count += 1
            N = voc.shape[0]
            voc = voc.cuda()
            l_voc = l_voc.cuda()
            G2 = self.H_record[iter_count,:N,:]

            # cof = None
            latent = self.CHEncoder(l_voc)

            tx = latent.view(N, self.P, self.S, 2, self.M).permute(0, 1, 2, 4, 3)

            out_pilot, out_sig, noise_pwr, self.PAPR, self.PAPR_cp = self.OFDM(
                self.normalize(tx, 1),
                K=self.K,
                SNR=self.SNR,
                cof=G2)

            N, C, H, W = latent.shape

            if self.feedforward == 'IMPLICIT':
                r1 = self.OFDM.pilot.repeat(N, self.P, 1, 1, 1)
                r2 = out_pilot
                r3 = out_sig
                dec_in = torch.cat((r1, r2, r3), 2).contiguous().permute(0, 1, 2, 4, 3).contiguous().view(N, -1, H,
                                                                                                          W)
                fake = self.Generator(dec_in)
                # fake = self.SRGenerator(restore)
            elif self.feedforward == 'EXPLICIT-CE':
                # Channel estimation
                if self.CE == 'LS':
                    self.H_est = LS_channel_est(self.OFDM.pilot, out_pilot)
                elif self.CE == 'LMMSE':
                    self.H_est = LMMSE_channel_est(self.OFDM.pilot, out_pilot, self.M * noise_pwr)
                elif self.CE == 'TRUE':
                    self.H_est = self.H_true.unsqueeze(2).to(self.device)
                else:
                    raise NotImplementedError('The channel estimation method [%s] is not implemented' % self.CE)
                r1 = self.H_est
                r2 = out_pilot
                r3 = out_sig
                dec_in = torch.cat((r1, r2, r3), 2).contiguous().permute(0, 1, 2, 4, 3).contiguous().view(N, -1, H,
                                                                                                          W)
                fake = self.Generator(dec_in)
            elif self.feedforward == 'EXPLICIT-RES':
                # Channel estimation
                if self.CE == 'LS':
                    self.H_est = LS_channel_est(self.OFDM.pilot, out_pilot)
                elif self.CE == 'LMMSE':
                    self.H_est = LMMSE_channel_est(self.OFDM.pilot, out_pilot, self.M * noise_pwr)
                elif self.CE == 'TRUE':
                    self.H_est = self.H_true.unsqueeze(2).to(self.device)
                else:
                    raise NotImplementedError('The channel estimation method [%s] is not implemented' % self.CE)

                sub11 = self.OFDM.pilot.repeat(N, 1, 1, 1, 1)
                sub12 = out_pilot
                sub1_input = torch.cat((sub11, sub12), 2).contiguous().permute(0, 1, 2, 4, 3).contiguous().view(N,
                                                                                                                -1,
                                                                                                                H,
                                                                                                                W)
                sub1_output = self.SubNet1(sub1_input)
                sub1_output = sub1_output.view(N, self.P, 1, 2, self.M).permute(0, 1, 2, 4, 3)
                # self.equalization(self.H_est + sub1_output, out_sig, noise_pwr)

                # Equalization
                if self.EQ == 'ZF':
                    self.rx = ZF_equalization(self.H_est, out_sig)
                elif self.EQ == 'MMSE':
                    self.rx = MMSE_equalization(self.H_est, out_sig, self.config.M * noise_pwr)
                elif self.EQ == 'None':
                    self.rx = None
                else:
                    raise NotImplementedError('The equalization method [%s] is not implemented' % self.CE)

                sub21 = self.H_est + sub1_output
                sub22 = out_sig
                sub2_input = torch.cat((sub21, sub22), 2).contiguous().permute(0, 1, 2, 4, 3).contiguous().view(N,
                                                                                                                -1,
                                                                                                                H,
                                                                                                                W)
                sub2_output = self.SubNet2(sub2_input)
                sub2_output = sub2_output.view(N, self.P, self.S, 2, self.M).permute(0, 1, 2, 4, 3)
                dec_in = (self.rx + sub2_output).permute(0, 1, 2, 4, 3).contiguous().view(latent.shape)
                fake = self.Generator(dec_in)
                # fake = self.SRGenerator(restore)

            if (N == self.batch_size):
                batch_fake = fake
                batch_src = voc
                batch_low = l_voc

            mse = self.MSE(fake, voc)
            if (mse > 0):
                psnr_low2high = 10 * np.log10(
                    np.max(fake.cpu().detach().numpy()) ** 2 / mse.cpu().detach().numpy() ** 2)
                sum_psnr_low2high += psnr_low2high
            # update D
            # self.loss_D_fake,self.loss_D_real = self.update_discriminator()
            if self.gan_mode != 'none':
                self.optimizer_D.zero_grad()  # set D's gradients to zero
                # calculate gradients for D
                self.set_requires_grad(self.Discriminator, True)  # enable backprop for D
                _, pred_fake = self.Discriminator(fake.detach())
                self.loss_D_fake = self.criterionGAN(pred_fake, False)
                _, pred_real = self.Discriminator(voc)
                self.loss_D_real = self.criterionGAN(pred_real, True)

                if self.gan_mode in ['lsgan', 'vanilla']:
                    self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
                    self.loss_D.backward()
                elif self.gan_mode == 'wgangp':
                    penalty, grad = cal_gradient_penalty(self.Discriminator, voc, fake.detach(), self.device,
                                                         type='mixed', constant=1.0, lambda_gp=10.0)
                    self.loss_D = self.loss_D_fake + self.loss_D_real + penalty
                    self.loss_D.backward(retain_graph=True)

                self.optimizer_D.step()  # update D's weights
                self.set_requires_grad(self.Discriminator, False)  # enable backprop for D
            # update G
            # self.set_requires_grad(self.Discriminator, False)  # enable backprop for D
            # self.set_requires_grad(self.Generator, True)  # D requires no gradients when optimizing G
            loss_G_GAN = 0
            loss_G_Feat = 0

            self.optimizer_G.zero_grad()  # set G's gradients to zero
            # calculate graidents for G
            if self.gan_mode != 'none':
                feat_fake, pred_fake = self.Discriminator(fake)
                self.loss_G_GAN = self.criterionGAN(pred_fake, False)

                if self.is_Feat:
                    feat_real, pred_real = self.Discriminator(voc)
                    self.loss_G_Feat = 0

                    for j in range(len(feat_real)):
                        self.loss_G_Feat += self.criterionFeat(feat_real[j].detach(),
                                                               feat_fake[j]) * self.config.lambda_feat
                else:
                    self.loss_G_Feat = 0
                self.loss_G_L2 = self.criterionL2(fake, voc) * self.lambda_L2
                self.loss_PAPR = torch.mean(self.PAPR_cp)
                self.loss_G = self.loss_G_GAN + self.loss_G_Feat + self.loss_G_L2
                self.loss_G.backward()
                self.optimizer_G.step()  # udpate G's weights

            sum_l_g_f += self.loss_G.item()
            sum_l_d_f += self.loss_D_fake.item()
            sum_l_d_t += self.loss_D_real.item()
            info = (
                        'VALIDATE PHASE Epoch [%d/%d], lr: %.8f, d_real_loss: %.6f, ' 'd_fake_loss: %.6f, g_loss: %.4f, psnr_low2high: %.4f'
                        % (epoch + 1, self.n_epochs + self.last_epoch, self.optimizers[0].state_dict()['param_groups'][0]['lr'], sum_l_d_t / iter_count, sum_l_d_f / iter_count,
                           sum_l_g_f / iter_count, sum_psnr_low2high / iter_count))

            tqdm_loader.set_description_str(info)

        g_loss = sum_l_g_f / iter_count
        if (self.best_g_loss_valid > g_loss):
            print("save best model")
            self.best_g_loss_valid = g_loss
            merged = self.merge_images(batch_low.cpu().detach().numpy(),
                                       batch_fake.cpu().detach().numpy())
            path = os.path.join(self.sample_path,
                                'sample-best-' + self.dataset + '-lh-%.2f.png'%(self.scale_rate) )
            # scipy.misc.imsave(path, merged)
            imageio.imwrite(path, merged)
            print('saved %s' % path)

            # high and fixed-high
            merged = self.merge_images(batch_src.cpu().detach().numpy(), batch_fake.cpu().detach().numpy())
            path = os.path.join(self.sample_path,
                                'sample-best-' + self.dataset + '-hh-%.2f.png'%(self.scale_rate))
            # scipy.misc.imsave(path, merged)
            imageio.imwrite(path, merged)
            print('saved %s' % path)
            # save saved_models
            # save the model parameters for each epoch
            CHEncoder_path = os.path.join(self.model_path,
                                          self.dataset + 'best_CHEncoder_%.2f.pkl'%(self.scale_rate))
            Generator_path = os.path.join(self.model_path,
                                          self.dataset + 'best_Generator_%.2f.pkl'%(self.scale_rate))
            Discriminator_path = os.path.join(self.model_path,
                                              self.dataset + 'best_Discriminator_%.2f.pkl'%(self.scale_rate))
            torch.save(self.CHEncoder.state_dict(), CHEncoder_path)
            torch.save(self.Generator.state_dict(), Generator_path)
            torch.save(self.Discriminator.state_dict(), Discriminator_path)

    def record_H(self):
        self.MISO = CreateMISOEnv(8, 3, antenna_num=self.L, unit_num=40, r_min=3, bw=100, transmit_p_max=5,
                                  mec_p_max=38, with_CoMP=True, open_matlab=False, mec_rule="default")

        epoch_H_record = []
        self.MISO.reset()
        H_record = []
        for i in tqdm(range(1, 5000)):
            G2 = self.MISO.step(i)
            H_record.append(G2)
            epoch_H_record.append(H_record)
        epoch_H_record = np.array(epoch_H_record)
        np.savez("record_H_record", H=epoch_H_record)

    def name(self):
        return 'JSCCOFDM_Model'







