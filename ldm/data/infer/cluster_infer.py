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
        self.resize_ref = T.Resize([224,224])
        
        self.img_paths = self.args['dataset_dir']
        self.img_list = os.listdir(self.img_paths)
        
        self.length = len(self.img_list)
        self.img_list.sort()
 

       

    
    def __getitem__(self, index):
        img_path= self.img_paths+"/"+self.img_list[index]
        img_p = Image.open(img_path).convert("RGB")

        ### Generate mask
        image_tensor = get_tensor()(img_p)
        
        ref_tensor = self.resize_ref(get_tensor_clip()(img_p))

        image_tensor_resize=self.resize(image_tensor)
        real_mask_tensor_resize = torch.zeros_like(image_tensor_resize)[0:1,:,:]
        
        temp = 1 - real_mask_tensor_resize
        inpaint_tensor_resize=image_tensor_resize*temp

        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":temp,"real_mask":real_mask_tensor_resize,"ref_imgs":ref_tensor}



    def __len__(self):
        return self.length



