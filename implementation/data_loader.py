#!/usr/bin/env python
# encoding: utf-8
"""
@author: jiale xia
@contact: 2171953@neu.edu.cn
@file: data_loader_voc.py
@time: 2022/10/21 11:00
"""
import os
from  matplotlib import  pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.data as data
import glob
from PIL import Image
import  numpy as np


# def train_hr_transform(crop_size):
#     return Compose([
#         RandomCrop(crop_size),
#         ToTensor(),
#     ])

# def img_crop(img,crop_w,crop_h):
#
#         width, height = img.size
#
#         id = 1
#         i = 0
#         padw = padh = 0  # 当宽高除不尽切块大小时，对最后一块进行填充
#         if width % crop_w != 0:
#             padw = 1  # 宽除不尽的情况
#         if height % crop_h != 0:
#             padh = 1  # 高除不尽的情况
#
#         # 默认从最左上角向右裁剪，再向下裁剪
#         while i + crop_h <= height:
#             j = 0
#             while j + crop_w <= width:
#                 new_img = img.crop((j, i, j + crop_w, i + crop_h))
#                 # new_img.save(save_path + a + "_" + str(id) + b)
#                 id += 1
#                 j += crop_w
#             if padw == 1:  # 宽有除不尽的情况
#                 new_img = img.crop((width - crop_w, i, width, i + crop_h))
#                 # new_img.save(save_path + a + "_" + str(id) + b)
#                 id += 1
#             i = i + crop_h
#
#         if padh == 1:  # 高除不尽的情况
#             j = 0
#             while j + crop_w <= width:
#                 new_img = img.crop((j, height - crop_h, j + crop_w, height))
#                 # new_img.save(save_path + a + "_" + str(id) + b)
#                 id += 1
#                 j += crop_w
#             if padw == 1:
#                 new_img = img.crop((width - crop_w, height - crop_h, width, height))
#                 # new_img.save(save_path + a + "_" + str(id) + b)
#                 id += 1
#         return  new_img
# def train_lr_transform(crop_size, upscale_factor):
#     return Compose([
#         ToPILImage(),
#         Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
#         ToTensor()
#     ])

def getDataset(config):
    train_dataloaders, val_dataloaders ,test_dataloaders= None,None,None
    x_transforms = transforms.Compose([
        # transforms.Resize([256,256]),
        transforms.RandomCrop((config.crop_w,config.crop_h)),
        transforms.ToTensor(),  # -> [0,1]
        # transforms.RandomRotation(45),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    # mask只需要转换为tensor
    y_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([int(config.crop_w * config.scale_rate),int(config.crop_h * config.scale_rate) ], interpolation=Image.BILINEAR),
        transforms.Resize([config.crop_w, config.crop_h], interpolation=Image.EXTENT),
        # transforms.Resize([480,270]),
        transforms.ToTensor(),  # -> [0,1]
    ])
    if config.dataset =='VOC':  #E:\代码\new\u_net_liver-master\data\liver\val
        config.state="train"
        train_dataset = VOCDataset(config, x_transform=x_transforms, y_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=config.batch_size)
        config.state="val"
        val_dataset = VOCDataset(config, x_transform=x_transforms, y_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=config.batch_size)
        test_dataloaders = val_dataloaders
    if config.dataset =="CoCo":
        config.state="train"
        train_dataset = CoCoDataset(config, x_transform=x_transforms, y_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=config.batch_size)
        config.state="val" 
        val_dataset = CoCoDataset(config, x_transform=x_transforms, y_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=config.batch_size)
        test_dataloaders = val_dataloaders
    # if config.dataset == "dsb2018Cell":
    #     train_dataset = dsb2018CellDataset(r"train", transform=x_transforms, target_transform=y_transforms)
    #     train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
    #     val_dataset = dsb2018CellDataset(r"val", transform=x_transforms, target_transform=y_transforms)
    #     val_dataloaders = DataLoader(val_dataset, batch_size=1)
    #     test_dataloaders = val_dataloaders
    # if config.dataset == 'corneal':
    #     train_dataset = CornealDataset(r'train',transform=x_transforms, target_transform=y_transforms)
    #     train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
    #     val_dataset = CornealDataset(r"val", transform=x_transforms, target_transform=y_transforms)
    #     val_dataloaders = DataLoader(val_dataset, batch_size=1)
    #     test_dataset = CornealDataset(r"test", transform=x_transforms, target_transform=y_transforms)
    #     test_dataloaders = DataLoader(test_dataset, batch_size=1)
    # if config.dataset == 'driveEye':
    #     train_dataset = DriveEyeDataset(r'train', transform=x_transforms, target_transform=y_transforms)
    #     train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
    #     val_dataset = DriveEyeDataset(r"val", transform=x_transforms, target_transform=y_transforms)
    #     val_dataloaders = DataLoader(val_dataset, batch_size=1)
    #     test_dataset = DriveEyeDataset(r"test", transform=x_transforms, target_transform=y_transforms)
    #     test_dataloaders = DataLoader(test_dataset, batch_size=1)
    # if config.dataset == 'isbiCell':
    #     train_dataset = IsbiCellDataset(r'train', transform=x_transforms, target_transform=y_transforms)
    #     train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
    #     val_dataset = IsbiCellDataset(r"val", transform=x_transforms, target_transform=y_transforms)
    #     val_dataloaders = DataLoader(val_dataset, batch_size=1)
    #     test_dataset = IsbiCellDataset(r"test", transform=x_transforms, target_transform=y_transforms)
    #     test_dataloaders = DataLoader(test_dataset, batch_size=1)
    # if config.dataset == 'kaggleLung':
    #     train_dataset = LungKaggleDataset(r'train', transform=x_transforms, target_transform=y_transforms)
    #     train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
    #     val_dataset = LungKaggleDataset(r"val", transform=x_transforms, target_transform=y_transforms)
    #     val_dataloaders = DataLoader(val_dataset, batch_size=1)
    #     test_dataset = LungKaggleDataset(r"test", transform=x_transforms, target_transform=y_transforms)
    #     test_dataloaders = DataLoader(test_dataset, batch_size=1)
    # if config.dataset == 'glass':
    #     train_dataset = GlassDataset(r'train', transform=x_transforms, target_transform=y_transforms)
    #     train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
    #     val_dataset = GlassDataset(r"val", transform=x_transforms, target_transform=y_transforms)
    #     val_dataloaders = DataLoader(val_dataset, batch_size=1)
    #     test_dataset = GlassDataset(r"test", transform=x_transforms, target_transform=y_transforms)
    #     test_dataloaders = DataLoader(test_dataset, batch_size=1)

    return train_dataloaders,val_dataloaders,test_dataloaders


class VOCDataset(data.Dataset):
    def __init__(self,config,x_transform,y_transform):
        self.state = config.state
        self.train_root = config.train_root
        self.val_root = config.val_root
        self.test_root = config.test_root
        self.pics = self.getDataPath()
        self.reduce_resolution_scale = 0.2
        self.x_transform = x_transform
        self.y_transform = y_transform
    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root
        pics = glob.glob(os.path.join(root,"*.*"))
        # pics = []
        # n = len(os.listdir(root))
        # for i in range(n):
        #     img = os.path.join(root, "%03d.jpg" % i)  # liver is %03d
        #     pics.append(img)
        #     #imgs.append((img, mask))
        return pics

    def __getitem__(self, index):
        #x_path, y_path = self.imgs[index]
        origin_x = Image.open(self.pics[index])
        #origin_x = origin_x.convert(mode='RGB')
        # low_resolution_one = decrease_resolution_single(self.pics[index], self.reduce_resolution_scale)
        # origin_x = cv2.imread(x_path)
        # origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY)

        img_x = self.x_transform(origin_x)
        img_y=self.y_transform(img_x)

        # img_y=img_y.cpu().detach().numpy().transpose(1, 2, 0)
        # img_y = decrease_resolution_single(img_y, self.reduce_resolution_scale)
        # img_y = self.y_transform(img_y)
        # img_y = np.array(img_y)
        # img_x = np.array(img_x).transpose(1, 2, 0)
        # plt.figure()
        # plt.imshow(img_x)
        # plt.show()
        # plt.figure()
        # plt.imshow(img_y)
        # plt.show()
        return img_x, img_y

    def __len__(self):
        return len(self.pics)
        
class CoCoDataset(data.Dataset):
    def __init__(self,config,x_transform,y_transform):
        self.state = config.state
        self.train_root = config.train_root
        self.val_root = config.val_root
        self.test_root = config.test_root
        self.pics = self.getDataPath()
        self.reduce_resolution_scale = 0.2
        self.x_transform = x_transform
        self.y_transform = y_transform
    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root
        pics = glob.glob(os.path.join(root,"*.*"))
        # pics = []
        # n = len(os.listdir(root))
        # for i in range(n):
        #     img = os.path.join(root, "%03d.jpg" % i)  # liver is %03d
        #     pics.append(img)
        #     #imgs.append((img, mask))
        return pics

    def __getitem__(self, index):
        #x_path, y_path = self.imgs[index]
        origin_x = Image.open(self.pics[index])
        #if(origin_x.size[-1]==1):
        origin_x = origin_x.convert(mode='RGB')
        # low_resolution_one = decrease_resolution_single(self.pics[index], self.reduce_resolution_scale)
        # origin_x = cv2.imread(x_path)
        # origin_y = cv2.imread(y_path,cv2.COLOR_BGR2GRAY)

        img_x = self.x_transform(origin_x)
        img_y=self.y_transform(img_x)

        # img_y=img_y.cpu().detach().numpy().transpose(1, 2, 0)
        # img_y = decrease_resolution_single(img_y, self.reduce_resolution_scale)
        # img_y = self.y_transform(img_y)
        # img_y = np.array(img_y)
        # img_x = np.array(img_x).transpose(1, 2, 0)
        # plt.figure()
        # plt.imshow(img_x)
        # plt.show()
        # plt.figure()
        # plt.imshow(img_y)
        # plt.show()
        return img_x, img_y

    def __len__(self):
        return len(self.pics)


def decrease_resolution_multiple(image_dir_path,reduce_resolution_rate):
    # Get all png files under the input folder
    #image_dir_path=input/*.png
    input_img_path = glob.glob(os.path.join(image_dir_path,"*.*"))
    save_path = "low_resolution4x/"

    for file in input_img_path:
        # get the file_name of image
        file_name = file.split('\\')[-1]
        im = Image.open(file)
        image_size = im.size
        image_width = image_size[0]
        image_heigth = image_size[1]

        # reduce resolution by multiple
        # reduce_resolution_rate = 2
        new_image_width = image_width // reduce_resolution_rate
        new_image_height = image_heigth // reduce_resolution_rate

        # 降低图像分辨率reduce_resolution_rate倍
        im.thumbnail((new_image_width, new_image_height))
        # 恢复图像为原始大小，但是分辨率是降低后的
        im = im.transform(image_size, Image.EXTENT, (0, 0, new_image_width, new_image_height))

        # save reusult image to target folder
        im.save(save_path + file_name)

def decrease_resolution_single(image_path,reduce_resolution_scale):
    # Get all png files under the input folder
    #image_dir_path=input/*.png
    # get the file_name of image
    im = Image.open(image_path)
    image_size = im.size
    image_width = image_size[0]
    image_heigth = image_size[1]
    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    # reduce resolution by multiple
    # reduce_resolution_rate = 2
    new_image_width = int(image_width * reduce_resolution_scale)
    new_image_height = int(image_heigth * reduce_resolution_scale)

    # 降低图像分辨率reduce_resolution_rate倍
    im.thumbnail((new_image_width, new_image_height))
    # 恢复图像为原始大小，但是分辨率是降低后的
    im = im.transform(image_size, Image.ADAPTIVE, (0, 0, new_image_width, new_image_height))
    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    # save reusult image to target folder
    return im
def decrease_resolution_single_im(im,reduce_resolution_scale):
    # Get all png files under the input folder
    #image_dir_path=input/*.png
    # get the file_name of image
    # im = Image.open(image_path)
    image_size = im.size
    image_width = image_size[0]
    image_heigth = image_size[1]
    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    # reduce resolution by multiple
    # reduce_resolution_rate = 2
    new_image_width = int(image_width * reduce_resolution_scale)
    new_image_height = int(image_heigth * reduce_resolution_scale)

    # 降低图像分辨率reduce_resolution_rate倍
    im.thumbnail((new_image_width, new_image_height))
    # 恢复图像为原始大小，但是分辨率是降低后的
    im = im.transform(image_size, Image.ADAPTIVE, (0, 0, new_image_width, new_image_height))
    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    # save reusult image to target folder
    return im