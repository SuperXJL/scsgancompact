# -- coding: utf-8 --
import copy
import functools
import os
from collections import OrderedDict
from math import acos

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from components.base_blocks import conv_relu, get_norm_layer, init_net, get_scheduler, PAPR, GANLoss, \
    cal_gradient_penalty, Normalize, LS_channel_est, LMMSE_channel_est, ZF_equalization, MMSE_equalization, \
    trans_complex_to_real



# 定义的模型基类
class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def __init__(self, config):
        super().__init__()
        """Initialize the BaseModel class.

        Parameters:
            config (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, config)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.configimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.config = config  # 所有的配置都在opt中
        self.gpu_ids = config.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.configimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    def set_input(self, input):
        pass

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    # def setup(self, config):
    #     """Load and print networks; create schedulers
    #
    #     Parameters:
    #         config (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
    #     """
    #     if self.isTrain:
    #         self.schedulers = [get_scheduler(optimizer, config) for optimizer in self.configimizers]
    #     if not self.isTrain or config.continue_train:
    #         load_suffix = 'iter_%d' % config.load_iter if config.load_iter > 0 else config.epoch
    #         self.load_networks(load_suffix)
    #     self.print_networks(config.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()




    # helper saving function that can be used by subclasses
    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.6f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad



class base_resnet(nn.Module):
    def __init__(self,layer_num):
        super(base_resnet, self).__init__()
        if(layer_num == 18):
            self.model = models.resnet18(pretrained=True)
            # self.model.load_state_dict(torch.load('./resnet18-5c106cde.pth'))
        elif(layer_num==34):
            self.model = models.resnet34(pretrained=True)
            # self.model.load_state_dict(torch.load('./resnet34-333f7ec4.pth'))
        elif(layer_num == 50):
            self.model = models.resnet50(pretrained=True)
            # self.model.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
        elif(layer_num==101):
            self.model = models.resnet101(pretrained=True)
            # self.model.load_state_dict(torch.load('./resnet101-5d3b4d8f.pth'))
        elif(layer_num==152):
            self.model = models.resnet152(pretrained=True)
            # self.model.load_state_dict(torch.load('./resnet152-b121ed2d.pth'))
        #self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.model.fc= nn.Linear(self.model.fc.in_features, class_num)
    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        # x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        #x = self.model.layer3(x)
        #x = self.model.layer4(x)
        # x = self.model.avgpool(x)
        # x = x.view(x.size(0), x.size(1))
        # x = self.model.fc(x)

        return x



class MLP_MNIST(nn.Module):
    # classifier
    def __init__(self,in_ch):
        super(MLP_MNIST, self).__init__()
        self.fc1 = nn.Linear(in_ch, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 125)
        # self.fc4 = nn.Linear(125, 10)
        self.fc4 = nn.Linear(125, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class MLP(nn.Module):
    def __init__(self,in_ch, compression_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_ch, int(in_ch * compression_rate))
        self.fc2 = nn.Linear(int(in_ch * compression_rate),in_ch)

    def forward(self, x):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)

        # out_np = x.detach().cpu().numpy()
        # print(out_np)
        #
        # out_pd = np.array(out_np)

        # scale and quantize
        x = x.detach().cpu()
        x_max = torch.max(x)
        x_tmp = copy.deepcopy(torch.div(x, x_max))

        # quantize
        x_tmp = copy.deepcopy(torch.mul(x_tmp, 256))
        x_tmp = copy.deepcopy(x_tmp.clone().type(torch.int))
        x_tmp = copy.deepcopy(x_tmp.clone().type(torch.float32))
        x_tmp = copy.deepcopy(torch.div(x_tmp, 256))

        x = copy.deepcopy(torch.mul(x_tmp, x_max))

        aver_tmp = torch.mean(x, dim=0, out=None)
        aver = torch.mean(aver_tmp, dim=0, out=None)
        aver = abs(aver.item())

        snr = 3  # dB
        aver_noise = aver / 10 ** (snr / 10)
        noise = torch.randn(size=x.shape) * aver_noise

        x = x + noise

        # forward
        x_np = x.detach().cpu().numpy()
        out_square = np.square(x_np)
        aver = np.sum(out_square) / np.size(out_square)

        snr = 3  # dB
        aver_noise = aver / 10 ** (snr / 10)
        noise = np.random.random(size=x_np.shape) * np.sqrt(aver_noise)
        # noise = noise.to(device)

        x_np = x_np + noise
        x = torch.from_numpy(x_np)
        x = x.to(torch.float32)
        x = x.cuda()

        x = self.fc2(x)
        return x

class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(inception, self).__init__()
        # the first line
        self.branch1x1 = conv_relu(in_channel, out1_1, 1)

        # the second line
        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )

        # the thrid line
        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )

        # the fourth line
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)
        )

    def forward(self, x):
        # forward
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output


class googlenet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(googlenet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channels=64, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2)
        )
        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2)
        )
        self.block3 = nn.Sequential(
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2)
        )
        self.block4 = nn.Sequential(
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )
        self.block5 = nn.Sequential(
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(2)
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))

        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


class RED_CNN(nn.Module):
    def __init__(self, out_ch=16):
        super(RED_CNN, self).__init__()
        channel = 2
        self.conv1 = nn.Conv2d(3, out_ch, kernel_size=channel, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=channel, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 3, kernel_size=channel, stride=1, padding=0)

        # self.relu = nn.ReLU()

    def forward(self, x):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # encoder
        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.conv3(out)

        # scale and quantize
        out = out.detach().cpu()
        out_max = torch.max(out)
        out_tmp = copy.deepcopy(torch.div(out, out_max))

        # quantize
        out_tmp = copy.deepcopy(torch.mul(out_tmp, 256))
        out_tmp = copy.deepcopy(out_tmp.clone().type(torch.int))
        out_tmp = copy.deepcopy(out_tmp.clone().type(torch.float32))
        out_tmp = copy.deepcopy(torch.div(out_tmp, 256))

        out = copy.deepcopy(torch.mul(out_tmp, out_max))

        out_tmp = out.detach().cpu().numpy()
        out_square = np.square(out_tmp)
        aver = np.sum(out_square) / np.size(out_square)

        snr = 3  # dB
        aver_noise = aver / 10 ** (snr / 10)
        noise = torch.randn(size=out.shape) * np.sqrt(aver_noise)
        noise = noise.to(device)

        out = out + noise
        # out = torch.from_numpy(out)
        # out = out.to(torch.float32)
        # out = out.to(device)

        # print('out_after:', out.shape)

        # decoder
        # print('out_4:', out.shape)
        # out = self.tconv3(out)
        out = self.tconv4(out)
        out = self.tconv5(out)
        # print('out_5:', out.shape)
        # out += residual_1
        # out = self.relu(out)
        # print('shape of out:', out.size())
        return out


class coding(nn.Module):
    def __init__(self,d_conv_kernel ,out_ch=16):
        super(coding, self).__init__()
        # channel = 2
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=d_conv_kernel, stride=1, padding=0)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=d_conv_kernel, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=d_conv_kernel, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=d_conv_kernel, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=d_conv_kernel, stride=1, padding=0)
        self.conv_fixed1 = nn.Conv2d(3, out_ch, kernel_size=6, stride=1, padding=0)
        self.conv_fixed2 = nn.Conv2d(out_ch, out_ch, kernel_size=6, stride=1, padding=0)
        self.conv_fixed3 = nn.Conv2d(out_ch, out_ch, kernel_size=6, stride=1, padding=0)
        self.conv_fixed4 = nn.Conv2d(out_ch, out_ch, kernel_size=6, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=d_conv_kernel, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=d_conv_kernel, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=d_conv_kernel, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=d_conv_kernel, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=d_conv_kernel, stride=1, padding=0)
        self.tconv_fixed1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=6, stride=1, padding=0)
        self.tconv_fixed2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=6, stride=1, padding=0)
        self.tconv_fixed3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=6, stride=1, padding=0)
        self.tconv_fixed4 = nn.ConvTranspose2d(out_ch, 3, kernel_size=6, stride=1, padding=0)

        self.pool = nn.MaxPool2d(2, stride=2)
        # self.unpool = nn.MaxUnpool2d(3, stride=3)
        self.unpool = nn.Upsample(scale_factor=2, mode='nearest')

        # self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        out = self.pool(x)  # 48*48
        out = self.conv_fixed1(out)  # 43*43
        out = self.conv_fixed2(out)  # 38*38
        out = self.conv_fixed3(out)  # 33*33
        out = self.conv_fixed4(out)  # 28*28
        out = self.conv1(out)
        out = self.conv2(out)
        # out = self.conv3(out)

        # scale and quantize
        out = out.detach().cpu()
        out_max = torch.max(out)
        out_tmp = copy.deepcopy(torch.div(out, out_max))

        # quantize
        out_tmp = copy.deepcopy(torch.mul(out_tmp, 256))
        out_tmp = copy.deepcopy(out_tmp.clone().type(torch.int))
        out_tmp = copy.deepcopy(out_tmp.clone().type(torch.float32))
        out_tmp = copy.deepcopy(torch.div(out_tmp, 256))

        out = copy.deepcopy(torch.mul(out_tmp, out_max))

        out_tmp = out.detach().cpu().numpy()
        out_square = np.square(out_tmp)
        aver = np.sum(out_square) / np.size(out_square)

        # snr = 3  # dB
        snr = 10  # dB
        aver_noise = aver / 10 ** (snr / 10)
        noise = torch.randn(size=out.shape) * np.sqrt(aver_noise)
        noise = noise.cuda()

        out = out.cuda() + noise
        out = self.tconv4(out)
        out = self.tconv5(out)
        out = self.tconv_fixed1(out)
        out = self.tconv_fixed2(out)
        out = self.tconv_fixed3(out)
        out = self.tconv_fixed4(out)
        out = self.unpool(out)
        return out

class classifier_svhn(nn.Module):
    def __init__(self, out_ch=16):
        super(classifier_svhn, self).__init__()
        self.cla1 = nn.Conv2d(3, out_ch, kernel_size=5, stride=1, padding=0)  # 28*28
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)  # 14*14
        self.cla2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)  # 10*10
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)  # 5*5
        self.cla3 = nn.Conv2d(out_ch, 10, kernel_size=5, stride=1, padding=0)  # 1*1
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, out_ch, kernel_size=7, stride=1, padding=0)  # 26*26
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=7, stride=1, padding=0)  # 20*20
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=7, stride=1, padding=0)  # 14*14
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=7, stride=1, padding=0)  # 8*8
        self.conv5 = nn.Conv2d(out_ch, 10, kernel_size=8, stride=1, padding=0)  # 1*1

        self.cla = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))  # 16*16
        self.cla_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)  # 14*14
        self.cla_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)  # 12*12
        self.cla_conv2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)  # 10*10
        self.cla_conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)  # 8*8
        self.cla_conv2_3 = nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=0)  # 5*5
        self.cla_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)  # 3*3
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(32 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        # 分类器
        conv_out = self.cla(x)
        conv_out = self.cla_conv1(conv_out)
        conv_out = self.cla_conv2(conv_out)
        conv_out = self.cla_conv2_1(conv_out)
        conv_out = self.cla_conv2_2(conv_out)
        conv_out = self.cla_conv2_3(conv_out)
        conv_out = self.cla_conv3(conv_out)
        # print('shape of cla:', conv_out.size())
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)

        # out = self.cla1(x)  # 28*28
        # out = self.pool(out)  # 14*14
        # out = self.cla2(out)  # 10*10
        # out = self.pool(out)  # 5*5
        # out = self.cla3(out)  # 1*1
        # out = self.relu(out)

        # out = self.conv1(x)
        # out = self.conv2(out)
        # out = self.conv3(out)
        # out = self.conv4(out)
        # out = self.conv2(out)
        return out


##############################################################################
# Encoder
##############################################################################
class Encoder(nn.Module):

    def __init__(self, input_nc, ngf=64, max_ngf=512, input_channel=16, n_blocks=2, n_downsampling=2,
                 norm_layer=nn.BatchNorm2d, padding_type="reflect", first_kernel=7, first_add_C=0):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            ngf (int)           -- the number of filters in the first conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_downsampling >= 0)
        assert (n_blocks >= 0)
        super(Encoder, self).__init__()
        self.extractor = base_resnet(50)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d


        model = [nn.ReflectionPad2d((first_kernel - 1) // 2),
                 nn.Conv2d(input_nc, ngf, kernel_size=first_kernel, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # add downsampling layers
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(min(ngf * mult, max_ngf), min(ngf * mult * 2, max_ngf), kernel_size=3, stride=2, padding=1,
                          bias=use_bias),
                norm_layer(min(ngf * mult * 2, max_ngf)),
                nn.ReLU()]

        self.model_down = nn.Sequential(*model)
        model = []
        # add ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ ResnetBlock(min(ngf * mult, max_ngf) + first_add_C, padding_type=padding_type, norm_layer=norm_layer,
                            use_dropout=False, use_bias=use_bias)]

        self.model_res = nn.Sequential(*model)
        
        model = [
                 nn.Conv2d(512, input_channel, kernel_size=3, padding=1, bias=use_bias),
               ]
                 
        self.projection = nn.Sequential(*model)       

        #self.projection = nn.Conv2d(1024, input_channel, kernel_size=3, padding=1,
                                   # stride=1, bias=use_bias)

    def forward(self, imgae_data, H=None):
        z = self.extractor(imgae_data)
        #z = self.model_down(z)
        # if H is not None:
        #     n, c, h, w = z.shape
        #     #trans_H = H.contiguous().permute(0, 1, 2, 4, 3).reshape(n, -1, h, w)
        #     trans_H = H.contiguous().reshape(n, -1, h, w)
        #     z = torch.cat((z, trans_H), 1)
        #z = self.model_res(z)
        z = self.projection(z)
        return z


##############################################################################
# Generator
##############################################################################
class Generator(nn.Module):
    def __init__(self, output_nc, ngf=64, max_ngf=512, input_channel=16, n_downsampling=2,
                 norm_layer=nn.BatchNorm2d, padding_type="reflect", first_kernel=7, activation_='sigmoid'):
        assert (n_downsampling >= 0)

        super(Generator, self).__init__()

        self.activation_ = activation_

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        if activation_ == 'tanh':
            activation = nn.Tanh()
        elif activation_ == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_ == 'relu':
            activation = nn.ReLU(True)


        mult = 2 ** n_downsampling
        ngf_dim = min(ngf * mult, max_ngf)
        model = [nn.Conv2d(input_channel, ngf_dim, kernel_size=3, padding=1, stride=1, bias=use_bias)]

        for i in range(n_downsampling):
            model += [ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                                  use_bias=use_bias)]
                                  

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(ngf * mult, max_ngf), min(ngf * mult // 2, max_ngf),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(min(ngf * mult // 2, max_ngf)),
                      activation]
                      

        model += [nn.ReflectionPad2d((first_kernel - 1) // 2),
                  nn.Conv2d(ngf, output_nc, kernel_size=first_kernel, padding=0),
                  activation
                  ]

        #if activation_ == 'tanh':
            #model += [nn.Tanh()]
        #elif activation_ == 'sigmoid':
           # model += [nn.Sigmoid()]
        #elif activation_ == 'relu':
           # model += [activation]
        

        self.model = nn.Sequential(*model)

    def forward(self, input):

        if self.activation_ == 'tanh' or  self.activation_ == 'relu':
            return self.model(input)
        elif self.activation_ == 'sigmoid':
            return 2 * self.model(input) - 1
        
##############################################################################
# SRGenerator
##############################################################################
class SRGenerator(nn.Module):
    def __init__(self, output_nc, ngf=64, max_ngf=512, input_channel=16, n_blocks=2, n_downsampling=2,
                 norm_layer=nn.BatchNorm2d, padding_type="reflect", activation_='sigmoid'):
        assert (n_downsampling >= 0)

        super(SRGenerator, self).__init__()

        self.activation_ = activation_

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        if activation_ == 'tanh':
            activation = nn.Tanh()
        elif activation_ == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_ == 'relu':
            activation = nn.ReLU(True)


        mult = 2 ** n_downsampling
        ngf_dim = min(ngf * mult, max_ngf)
        model = [nn.Conv2d(input_channel, ngf_dim, kernel_size=3, padding=1, stride=1, bias=use_bias)]
        kernal = 5
        for i in range(n_downsampling):
            model += [
            nn.Conv2d(ngf_dim, ngf_dim, kernel_size=kernal, stride=2,padding=kernal//2),
            activation
          ]
        #  ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(ngf * mult, max_ngf), min(ngf * mult // 2, max_ngf),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(min(ngf * mult // 2, max_ngf)),
                      activation]

        model += [nn.ReflectionPad2d((kernal - 1) // 2),
                  nn.Conv2d(ngf, output_nc, kernel_size=kernal, padding=0),
                  activation
                  ]

        #if activation_ == 'tanh':
            #model += [nn.Tanh()]
        #elif activation_ == 'sigmoid':
           # model += [nn.Sigmoid()]
        #elif activation_ == 'relu':
           # model += [activation]
        

        self.model = nn.Sequential(*model)

    def forward(self, input):

        if self.activation_ == 'tanh' or  self.activation_ == 'relu':
            return self.model(input)
        elif self.activation_ == 'sigmoid':
            return 2 * self.model(input) - 1


#########################################################################################
# Residual block
#########################################################################################
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        self.n_layers = n_layers

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [
            [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]]  # output 1 channel prediction map

        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        """Standard forward."""
        res = [input]
        for n in range(self.n_layers + 1):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))

        model = getattr(self, 'model' + str(self.n_layers + 1))
        out = model(res[-1])

        return res[1:], out


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


#########################################################################################
# Residual-like subnet
#########################################################################################
class Subnet(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, dim_out, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(Subnet, self).__init__()
        self.conv_block = self.build_conv_block(dim, dim_out, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, dim_out, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
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

        conv_block += [nn.Conv2d(dim, 64, kernel_size=3, padding=p, bias=use_bias), norm_layer(64), nn.ReLU(True)]
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
        conv_block += [nn.Conv2d(64, dim_out, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim_out)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = self.conv_block(x)  # add skip connections
        return out




class BatchConv1DLayer(nn.Module):
    def __init__(self, stride=1,
                 padding=0, dilation=1):
        super(BatchConv1DLayer, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x, weight, bias=None):
        if bias is None:
            assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
        else:
            assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
                0], "dim=0 of bias must be equal in size to dim=0 of weight"

        b_i, b_j, c, h = x.shape
        b_i, out_channels, in_channels, kernel_width_size = weight.shape

        out = x.permute([1, 0, 2, 3]).contiguous().view(b_j, b_i * c, h)
        weight = weight.view(b_i * out_channels, in_channels, kernel_width_size)

        out = F.conv1d(out, weight=weight, bias=None, stride=self.stride, dilation=self.dilation, groups=b_i,
                       padding=self.padding)

        out = out.view(b_j, b_i, out_channels, out.shape[-1])

        out = out.permute([1, 0, 2, 3])

        if bias is not None:
            out = out + bias.unsqueeze(1).unsqueeze(3)

        return out


# Clipping layer
def Clipping(x,CR):
    # Calculate the additional non-linear noise
    amp = torch.sqrt(torch.sum(x ** 2, -1, True))
    sigma = torch.sqrt(torch.mean(x ** 2, (-2, -1), True) * 2)
    ratio = sigma * CR / amp
    scale = torch.min(ratio, torch.ones_like(ratio))

    with torch.no_grad():
        bias = x * scale - x

    return x + bias

def Add_CP(x,k):
    return torch.cat((x[..., -k:, :], x), dim=-2)

def Rm_CP(x,k):
    return x[..., k:, :]

# Realization of multipath channel as a nn module
class Channel(nn.Module):
    def __init__(self, config, device):
        super(Channel, self).__init__()

        self.config = config
        # Generate unit power profile
        power = torch.exp(-torch.arange(config.L).float() / config.decay).view(1, 1, config.L, 1)  # 1x1xLx1
        self.power = power / torch.sum(power)  # Normalize the path power to sum to 1
        self.device = device

        # Initialize the batched 1d convolution layer
        self.bconv1d = BatchConv1DLayer(padding=config.L - 1)

    # def sample(self, N, P, M, L):
    #     # Sample the channel coefficients
    #     cof = torch.sqrt(self.power / 2) * torch.randn(N, P, L, 2)
    #     cof_zp = torch.cat((cof, torch.zeros((N, P, M - L, 2))), 2)
    #     fft_rs = torch.fft.fft(cof_zp, dim=1)
    #     #H_t =torch.cat([fft_rs.real,fft_rs.imag])
    #
    #     H_t =trans_complex_to_real(fft_rs)
    #     # H_t =0.5*(fft_rs.real+fft_rs.imag)
    #     return cof, H_t

    def forward(self, input, cof=None):
        # Input size:   NxPx(Sx(M+K))x2
        # Output size:  NxPx(L+Sx(M+K)-1)x2
        # Also return the true channel
        # Generate Channel Matrix

        N, P, SMK, _ = input.shape

        # If the channel is not given, random sample one from the channel model
        # if cof is None:
        #     cof, H_t = self.sample(N, P, self.config.M, self.config.L)
        # else:
            # cof_zp = torch.cat((cof, torch.zeros((N, P, self.config.M - self.config.L, 2))), 2)
            # fft_rs = torch.fft.fft(cof_zp, dim=1)
            # H_t = trans_complex_to_real(fft_rs)
        # signal_real = input[..., 0].view(N * P, 1, 1, -1)  # (NxP)x1x1x(Sx(M+K))
        # signal_imag = input[..., 1].view(N * P, 1, 1, -1)  # (NxP)x1x1x(Sx(M+K))
        signal_real = input[..., 0].view(N * P, 1, 1, -1)  # (NxP)x1x1x(Sx(M+K))
        signal_imag = input[..., 1].view(N * P, 1, 1, -1)  # (NxP)x1x1x(Sx(M+K))

        ind = torch.linspace(self.config.L - 1, 0, self.config.L).long()
        cof_real = cof[..., 0][..., ind].view(N * P, 1, 1, -1).to(self.device)  # (NxP)x1x1xL
        cof_imag = cof[..., 1][..., ind].view(N * P, 1, 1, -1).to(self.device)  # (NxP)x1x1xL
        # cof_real = cof.real[..., ind].view(N * P, 1, 1, -1).to(self.device)  # (NxP)x1x1xL
        # cof_imag = cof.imag[..., ind].view(N * P, 1, 1, -1).to(self.device)  # (NxP)x1x1xL

        output_real = self.bconv1d(signal_real, cof_real) - self.bconv1d(signal_imag, cof_imag)  # (NxP)x1x1x(L+SMK-1)
        output_imag = self.bconv1d(signal_real, cof_imag) + self.bconv1d(signal_imag, cof_real)  # (NxP)x1x1x(L+SMK-1)

        output = torch.cat((output_real.view(N * P, -1, 1), output_imag.view(N * P, -1, 1)), -1)
        output = output.view(N, P,self.config.L + SMK - 1,2)  # NxPx(L+SMK-1)x2

        return output


# Realization of OFDM system as a nn module
# 有关OFDM信号处理技术的介绍参考 https://blog.csdn.net/weixin_43935696/article/details/108041325
class OFDM(nn.Module):
    def __init__(self, config,channel, pilot_path):
        super(OFDM, self).__init__()
        self.S = config.S
        self.K = config.K
        self.P =config.P
        self.M = config.M
        self.N_pilot=config.N_pilot
        self.CR = config.CR
        self.is_clip = config.is_clip
        # Setup the channel layer

        self.channel = channel
        # Generate the pilot signal
        if not os.path.exists(pilot_path):
            bits = torch.randint(2, (config.M, 2))
            torch.save(bits, pilot_path)
            pilot = (2 * bits - 1).float()
        else:
            bits = torch.load(pilot_path)
            pilot = (2 * bits - 1).float()

        self.pilot = pilot.cuda()
        ifft_rs = torch.fft.ifft(self.pilot,dim=1).real
        
        #ifft_rs = torch.cat([ifft_rs.real,ifft_rs.imag],1)
        # ifft_rs = ifft_rs.real
        self.pilot_cp = Add_CP(ifft_rs,self.K).repeat(config.P, config.N_pilot, 1, 1)
        #print(self.pilot_cp)

    def forward(self, x,K,cof,SNR, batch_size=None):
        # Input size: NxPxSxMx2   The information to be transmitted
        # cof denotes given channel coefficients

        # If x is None, we only send the pilots through the channel


        if x != None:
            N = x.shape[0]
            # x = x[:,0] +x[:,1]*1j

            # IFFT:                    NxPxSxMx2  => NxPxSxMx2
            x = torch.fft.ifft(x, dim=1).real
            # x = ifft_rs.real

            # Add Cyclic Prefix:       NxPxSxMx2  => NxPxSx(M+K)x2
            x = Add_CP(x,K)

            # Add pilot:               NxPxSx(M+K)x2  => NxPx(S+1)x(M+K)x2
            pilot = self.pilot_cp.repeat(N, 1, 1, 1,1)
            x = torch.cat((pilot, x), 2)
            Ns = self.S
        else:
            N = batch_size
            x = self.pilot_cp.repeat(N, 1, 1, 1, 1)
            Ns = 0

        # Reshape:                 NxPx(S+1)x(M+K)x2  => NxPx(S+1)(M+K)x2
        # x = gen_trans_h(x,N,(Ns + self.N_pilot),self.M+ self.K)
        x = x.view(N, self.P, (Ns + self.N_pilot) * (self.M + self.K), 2)

        papr = PAPR(x)

        # Clipping (Optional):     NxPx(S+1)(M+K)x2  => NxPx(S+1)(M+K)x2
        if self.is_clip:
            x = Clipping(x,self.CR)

        papr_cp = PAPR(x)

        # Pass through the Channel:        NxPx(S+1)(M+K)x2  =>  NxPx((S+1)(M+K)+L-1)x2
        y= self.channel(x, cof)

        # Calculate the power of received signal

        pwr = torch.mean(y ** 2, (-2, -1), True) * 2
        noise_pwr = pwr * 10 ** (-SNR / 10)

        # Generate random noise
        noise = torch.sqrt(noise_pwr / 2) * torch.randn_like(y)
        y_noisy = y + noise

        # Peak Detection: (Perfect)    NxPx((S+S')(M+K)+L-1)x2  =>  NxPx(S+S')x(M+K)x2
        output = y_noisy[:, :, :(Ns + self.N_pilot) * (self.M + self.K), :].view(N, self.P,Ns + self.N_pilot,self.M + self.K, 2)

        y_pilot = output[:, :, :self.N_pilot, :, :]  # NxPxS'x(M+K)x2
        y_sig = output[:, :, self.N_pilot:, :, :]  # NxPxSx(M+K)x2

        if  x != None:
            # Remove Cyclic Prefix:
            info_pilot = Rm_CP(y_pilot,K)  # NxPxS'xMx2
            info_sig = Rm_CP(y_sig,K)  # NxPxSxMx2
            # FFT:
            fft_rs_p = torch.fft.fft(info_pilot, dim=1)
            #info_pilot = torch.cat([fft_rs_p.real,fft_rs_p.imag],1)
            # info_pilot = 0.5*(fft_rs_p.real+fft_rs_p.imag)
            info_pilot =trans_complex_to_real(fft_rs_p)
            
            fft_rs_s = torch.fft.fft(info_sig, dim=1)
            
            #info_sig = torch.cat([fft_rs_s.real,fft_rs_s.imag],1)
            # info_sig = 0.5*(fft_rs_s.real+fft_rs_s.imag)
            info_sig = trans_complex_to_real(fft_rs_s)

            return info_pilot, info_sig, noise_pwr, papr, papr_cp
        else:
            info_pilot = Rm_CP(y_pilot,K)  # NxPxS'xMx2
            fft_rs_p = torch.fft.fft(info_pilot, dim=1)
            #info_pilot = torch.cat([fft_rs_p.real,fft_rs_p.imag],1)
            # info_pilot = 0.5*(fft_rs_p.real+fft_rs_p.imag)
            # info_pilot = 0.5 * (fft_rs_p.real + fft_rs_p.imag)
            info_pilot = trans_complex_to_real(fft_rs_p)

            return info_pilot, noise_pwr


# Realization of direct transmission over the multipath channel
class PLAIN(nn.Module):

    def __init__(self, config, device):
        super(PLAIN, self).__init__()
        self.config = config

        # Setup the channel layer
        self.channel = Channel(config, device)

    def forward(self, x, SNR):
        # Input size: NxPxMx2
        N, P, M, _ = x.shape
        y = self.channel(x, None)

        # Calculate the power of received signal
        pwr = torch.mean(y ** 2, (-2, -1), True) * 2
        noise_pwr = pwr * 10 ** (-SNR / 10)

        # Generate random noise
        noise = torch.sqrt(noise_pwr / 2) * torch.randn_like(y)
        y_noisy = y + noise  # NxPx(M+L-1)x2
        rx = y_noisy[:, :, :M, :]
        return rx

