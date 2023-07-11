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
    def __init__(self,state,**args
        ):
        self.state=state
        self.args=args
        self.kernel = np.ones((1, 1), np.uint8)

        self.resize = T.Resize([self.args['image_size'],self.args['image_size']])

        self.bbox_path_list=[]

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
        # real_mask = Image.open(real_mask_path).convert("L")
        
        ### Generate mask
        image_tensor = get_tensor()(img_p)
        # real_mask_tensor = T.ToTensor()(real_mask)
        W,H = img_p.size

        extended_bbox=copy.copy(bbox)
        left_freespace=bbox[0]-0
        right_freespace=W-bbox[2]
        up_freespace=bbox[1]-0
        down_freespace=H-bbox[3]
        
        extended_bbox[0]=bbox[0]-int(0.3*left_freespace)
        extended_bbox[1]=bbox[1]-int(0.3*up_freespace)
        extended_bbox[2]=bbox[2]+int(0.3*right_freespace)
        extended_bbox[3]=bbox[3]+int(0.3*down_freespace)

        mask_img=np.zeros((H,W))
        mask_img[extended_bbox[1]:extended_bbox[3],extended_bbox[0]:extended_bbox[2]]=1
        mask_img=Image.fromarray(mask_img)
        mask_tensor=get_tensor(normalize=False, toTensor=True)(mask_img)
       
        image_tensor_cropped=image_tensor
        real_mask_cropped = mask_tensor

        
        image_tensor_resize=self.resize(image_tensor_cropped)
        real_mask_tensor_resize=self.resize(real_mask_cropped)
        
        
        temp = 1 - real_mask_tensor_resize
        inpaint_tensor_resize=image_tensor_resize*temp

        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":temp,"real_mask":real_mask_tensor_resize}



    def __len__(self):
        return self.length




