from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image,ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import bezier
import pickle

def get_suitible(img_path, kernel_size = 3, iterations = 3):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.where((img >= 125) & (img <= 200), 255, 0)
    img = img.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    iterations = iterations
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CROSS, kernel, iterations=iterations)
    return opening
    
def get_boundary(img_path,kernel_size = 3, iterations = 3):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.where(img >= 7, 255, 0)
    img = img.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opening

def shift_mask(mask_1, mask_2, bbox):
    # 找到 mask_1 中值为 1 的位置
    indices = np.where(mask_1 == 1)
    if len(indices[0]) == 0:
        return mask_2
    # 随机选择一个值为 1 的位置作为平移参考点
    shift_row = 0
    shift_col = 0
    for i in range(3):
        index = np.random.choice(len(indices[0]))
        # 计算新 bbox 的中心点坐标
        shift_row += indices[0][index]//3
        shift_col += indices[1][index]//3

    # 计算平移后的 bbox 的左上角和右下角坐标
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    new_x1 = shift_col - bbox_width // 2
    new_y1 = shift_row - bbox_height // 2
    new_x2 = new_x1 + bbox_width
    new_y2 = new_y1 + bbox_height

    # 裁剪平移后的 bbox，确保不超出图像边界
    cols, rows = mask_2.shape
    new_x1 = max(new_x1, 0)
    new_y1 = max(new_y1, 0)
    new_x2 = min(new_x2, rows)
    new_y2 = min(new_y2, cols)

    new_rows, new_cols = new_y2 - new_y1, new_x2 - new_x1

    # 根据偏移量进行整体平移
    shifted_mask = np.zeros_like(mask_2)
    shifted_mask[new_y1:new_y2, new_x1:new_x2] = mask_2[bbox[1]:bbox[1] + new_rows, bbox[0]:bbox[0] + new_cols]

    return shifted_mask

def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


class OpenImageDataset(data.Dataset):
    def __init__(self,state, **args
        ):
        self.state=state
        self.args=args
        self.kernel = np.ones((5, 5), np.uint8)

        self.resize = T.Resize([self.args['image_size'],self.args['image_size']])
        self.resize_ref = T.Resize([224,224])
        self.bbox_path_list=[]
        
        self.augmentation = A.Compose([
            A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=(0, 0.05), border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ElasticTransform(p=0.6, alpha=1, sigma=50, alpha_affine=50, border_mode=cv2.BORDER_CONSTANT, value=0),
        ])

        bbox_dir = os.path.join(args['dataset_dir'], 'train_10_old', 'bboxs')       
        per_dir_file_list=os.listdir(bbox_dir)
        for file_name in per_dir_file_list:
            self.bbox_path_list.append(os.path.join(bbox_dir,file_name))
            
        self.bbox_path_list.sort()
        self.length=len(self.bbox_path_list)
        self.random = args['random']

        normal_dir = args['normal_dir']
        normal_files = os.listdir(normal_dir)
        normal_files.sort()
        self.normal_path_list = np.array([os.path.join(normal_dir, file_name) for file_name in normal_files])


    
    def __getitem__(self, index):
        bbox_path=self.bbox_path_list[index]
        file_name=os.path.splitext(os.path.basename(bbox_path))[0]+'.jpg'
        real_mask_path=os.path.join(self.args['dataset_dir'], self.state,'masks',file_name)
        normal_path = np.random.choice(self.normal_path_list)
        
        bbox_list = pickle.load(open(bbox_path, 'rb'))
        bbox = random.choice(bbox_list)
        
        real_mask = Image.open(real_mask_path).convert("L")
        img_p = Image.open(normal_path).convert("RGB")
        
        real_mask = np.asarray(real_mask)
        real_mask = cv2.morphologyEx(real_mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        avail = get_suitible(normal_path)//255
        not_background = get_boundary(normal_path)
        
        
        avail = A.Resize(real_mask.shape[0], real_mask.shape[1])(image=avail)['image']
        
        final_mask = shift_mask(avail, real_mask, bbox)
        if final_mask.sum() / (final_mask.shape[0]*final_mask.shape[1]) > 1/6:
            trans = A.ShiftScaleRotate(p=1, shift_limit=0.1, scale_limit=(-0.5, -0.2), border_mode=cv2.BORDER_CONSTANT, value=0)
            final_mask = trans(image=final_mask)['image']
            
        if self.random:
            seed = random.randint(0, 2 ** 32)
            np.random.seed(seed)
            final_mask = self.augmentation(image=final_mask)['image']
        

        
        final_mask = T.ToTensor()(final_mask)
        not_background = T.ToTensor()(not_background)
        
        
        
        final_mask = self.resize(final_mask)
        not_background = self.resize(not_background)
        
        
        final_mask = final_mask * not_background
        

        
        ### Generate mask
        image_tensor = get_tensor()(img_p)
        image_tensor_resize=self.resize(image_tensor)
        real_mask_tensor_resize= final_mask
        
        
        temp = 1 - real_mask_tensor_resize
        inpaint_tensor_resize=image_tensor_resize*temp

        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":temp,"real_mask":real_mask_tensor_resize}



    def __len__(self):
        return self.length



