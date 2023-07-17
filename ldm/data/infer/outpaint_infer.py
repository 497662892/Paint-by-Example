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
    def __init__(self,state,random=False,**args
        ):
        self.state=state
        self.args=args
        self.kernel = np.ones((1, 1), np.uint8)
        
        self.resize = T.Resize([self.args['image_size'],self.args['image_size']])
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.ElasticTransform(p=0.5)
            ])
        self.bbox_path_list=[]

        bbox_dir = os.path.join(args['dataset_dir'], 'train_10', 'bboxs')
        per_dir_file_list=os.listdir(bbox_dir)
        for file_name in per_dir_file_list:
            self.bbox_path_list.append(os.path.join(bbox_dir,file_name))
        self.bbox_path_list.sort()
        self.length=len(self.bbox_path_list)
        self.random=random

       

    
    def __getitem__(self, index):
        bbox_path=self.bbox_path_list[index]
        file_name=os.path.splitext(os.path.basename(bbox_path))[0]+'.jpg'
        img_path=os.path.join(self.args['dataset_dir'], self.state,'images',file_name)
        real_mask_path=os.path.join(self.args['dataset_dir'], self.state,'masks',file_name)

        img_p = cv2.imread(img_path)
        img_p = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)
        real_mask = cv2.imread(real_mask_path,0)

        
        if self.random:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            transformed  = self.augmentation(image = img_p, mask = real_mask)
            
        img_p = transformed['image']
        real_mask = transformed['mask']
        
        img_p = Image.fromarray(img_p)
        real_mask = Image.fromarray(real_mask)
                
        ### Generate mask
        image_tensor = get_tensor()(img_p)
        real_mask_tensor = T.ToTensor()(real_mask)

        image_tensor_resize=self.resize(image_tensor)
        real_mask_tensor_resize=self.resize(real_mask_tensor)
        
        



        temp = F.max_pool2d(real_mask_tensor_resize, kernel_size=5, stride=1, padding=2)
        inpaint_tensor_resize=image_tensor_resize*temp

        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":temp,"real_mask":real_mask_tensor_resize}



    def __len__(self):
        return self.length



