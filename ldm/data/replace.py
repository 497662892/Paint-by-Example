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
    def __init__(self,state,arbitrary_mask_percent=0,**args
        ):
        self.state=state
        self.args=args
        self.arbitrary_mask_percent=arbitrary_mask_percent
        self.kernel = np.ones((1, 1), np.uint8)
        self.random_trans=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3,border_mode=cv2.BORDER_CONSTANT),
            A.LongestMaxSize(max_size=224),
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT),
            ])
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.2,0.1), rotate_limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.ElasticTransform(p=0.5,border_mode=cv2.BORDER_CONSTANT),
        ])
        self.resize = T.Resize([self.args['image_size'],self.args['image_size']])

        self.bbox_path_list=[]
        if state == "train":
            bbox_dir = os.path.join(args['dataset_dir'], 'train_10', 'bboxs')
            per_dir_file_list=os.listdir(bbox_dir)
            for file_name in per_dir_file_list:
                self.bbox_path_list.append(os.path.join(bbox_dir,file_name))
        elif state == "val":
            bbox_dir = os.path.join(args['dataset_dir'], 'val', 'bboxs')
            per_dir_file_list=os.listdir(bbox_dir)
            for file_name in per_dir_file_list:
                self.bbox_path_list.append(os.path.join(bbox_dir,file_name))
        else:
            bbox_dir = os.path.join(args['dataset_dir'], 'train_10', 'bboxs')
            per_dir_file_list=os.listdir(bbox_dir)
            for file_name in per_dir_file_list:
                self.bbox_path_list.append(os.path.join(bbox_dir,file_name))
        self.bbox_path_list.sort()
        self.length=len(self.bbox_path_list)
 

       

    
    def __getitem__(self, index):
        bbox_path=self.bbox_path_list[index]
        file_name=os.path.splitext(os.path.basename(bbox_path))[0]+'.jpg'
        img_path=os.path.join(self.args['dataset_dir'], self.state,'images',file_name)
        real_mask_path=os.path.join(self.args['dataset_dir'], self.state,'masks',file_name)

        bbox_list=pickle.load(open(bbox_path,'rb'))
        bbox=random.choice(bbox_list)
        img_p = Image.open(img_path).convert("RGB")
        real_mask = Image.open(real_mask_path).convert("L")
        img_p = np.asarray(img_p, dtype=np.uint8)
        real_mask = np.asarray(real_mask,dtype=np.uint8)

        ### Get reference image
        bbox_pad=copy.copy(bbox)
        bbox_pad[0]=bbox[0]-min(10,bbox[0]-0)
        bbox_pad[1]=bbox[1]-min(10,bbox[1]-0)
        bbox_pad[2]=bbox[2]+min(10,img_p.shape[1]-bbox[2])
        bbox_pad[3]=bbox[3]+min(10,img_p.shape[0]-bbox[3])
        
        img_p_np=cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        ref_image_tensor=img_p_np[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2],:]
        
        ref_image_tensor=self.random_trans(image=ref_image_tensor)
        ref_image_tensor=Image.fromarray(ref_image_tensor["image"])
        
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)



        ### Generate mask
        if self.state=="train":
            seed = random.randint(0, 2 ** 32)
            random.seed(seed)
            np.random.seed(seed)
            trans = self.augmentation(image=img_p, mask=real_mask)
            img_p, real_mask = trans["image"],trans["mask"]
            
        image_tensor = get_tensor()(img_p)
        real_mask_tensor = T.ToTensor()(real_mask)


        image_tensor_cropped=image_tensor
        real_mask_cropped = real_mask_tensor

        
        image_tensor_resize=self.resize(image_tensor_cropped)
        real_mask_tensor_resize=self.resize(real_mask_cropped)        
        
        temp = 1 - real_mask_tensor_resize
        inpaint_tensor_resize=image_tensor_resize*temp

        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":temp,"ref_imgs":ref_image_tensor,"real_mask":real_mask_tensor_resize}



    def __len__(self):
        return self.length



