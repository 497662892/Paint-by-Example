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
            A.ElasticTransform(p=0.3),
            A.LongestMaxSize(max_size=224),
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT),
            ])
        self.augmentation = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])
        self.resize = T.Resize([self.args['image_size'],self.args['image_size']])

        self.images_path_list=[]
        if state == "train":
            images_dir = os.path.join(args['dataset_dir'], 'train')
            per_dir_file_list=os.listdir(images_dir)
            for file_name in per_dir_file_list:
                self.images_path_list.append(os.path.join(images_dir,file_name))
        elif state == "val":
            images_dir = os.path.join(args['dataset_dir'], 'val')
            per_dir_file_list=os.listdir(images_dir)
            for file_name in per_dir_file_list:
                self.images_path_list.append(os.path.join(images_dir,file_name))

        self.images_path_list.sort()
        self.length=len(self.images_path_list)




    
    def __getitem__(self, index):
        file_name=self.images_path_list[index]
        img_path=os.path.join(self.args['dataset_dir'], self.state,file_name)

        img_p = Image.open(img_path).convert("RGB")
        bbox_weight = np.random.uniform(0.3, 0.6)*img_p.size[0]
        bbox_height = np.random.uniform(0.3, 0.6)*img_p.size[1]
        bbox_left = np.random.uniform(0, 1-bbox_weight/img_p.size[0])*img_p.size[0]
        bbox_low = np.random.uniform(0, 1-bbox_height/img_p.size[1])*img_p.size[1]
        bbox_right = bbox_left + bbox_weight
        bbox_height = bbox_low + bbox_height
        
        bbox = [int(bbox_left), int(bbox_low), int(bbox_right), int(bbox_height)]

   
        ### Get reference image
        bbox_pad=copy.copy(bbox)

        img_p_np=cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        ref_image_tensor=img_p_np[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2],:]
        # ref_image_tensor = img_p_np

        ref_image_tensor=self.random_trans(image=ref_image_tensor)
        ref_image_tensor=Image.fromarray(ref_image_tensor["image"])
        
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)



        ### Generate mask
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

        extended_bbox=copy.copy(bbox)
        left_freespace=bbox[0]-0
        right_freespace=W-bbox[2]
        up_freespace=bbox[1]-0
        down_freespace=H-bbox[3]
        extended_bbox[0]=bbox[0]-random.randint(0,int(0.4*left_freespace))
        extended_bbox[1]=bbox[1]-random.randint(0,int(0.4*up_freespace))
        extended_bbox[2]=bbox[2]+random.randint(0,int(0.4*right_freespace))
        extended_bbox[3]=bbox[3]+random.randint(0,int(0.4*down_freespace))

        prob=random.uniform(0, 1)
        if prob<self.arbitrary_mask_percent:
            mask_img = Image.new('RGB', (W, H), (255, 255, 255)) 
            bbox_mask=copy.copy(bbox)
            extended_bbox_mask=copy.copy(extended_bbox)
            top_nodes = np.asfortranarray([
                            [bbox_mask[0],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[2]],
                            [bbox_mask[1], extended_bbox_mask[1], bbox_mask[1]],
                        ])
            down_nodes = np.asfortranarray([
                    [bbox_mask[2],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[0]],
                    [bbox_mask[3], extended_bbox_mask[3], bbox_mask[3]],
                ])
            left_nodes = np.asfortranarray([
                    [bbox_mask[0],extended_bbox_mask[0] , bbox_mask[0]],
                    [bbox_mask[3], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[1]],
                ])
            right_nodes = np.asfortranarray([
                    [bbox_mask[2],extended_bbox_mask[2] , bbox_mask[2]],
                    [bbox_mask[1], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[3]],
                ])
            top_curve = bezier.Curve(top_nodes,degree=2)
            right_curve = bezier.Curve(right_nodes,degree=2)
            down_curve = bezier.Curve(down_nodes,degree=2)
            left_curve = bezier.Curve(left_nodes,degree=2)
            curve_list=[top_curve,right_curve,down_curve,left_curve]
            pt_list=[]
            random_width=5
            for curve in curve_list:
                x_list=[]
                y_list=[]
                for i in range(1,19):
                    if (curve.evaluate(i*0.05)[0][0]) not in x_list and (curve.evaluate(i*0.05)[1][0] not in y_list):
                        pt_list.append((curve.evaluate(i*0.05)[0][0]+random.randint(-random_width,random_width),curve.evaluate(i*0.05)[1][0]+random.randint(-random_width,random_width)))
                        x_list.append(curve.evaluate(i*0.05)[0][0])
                        y_list.append(curve.evaluate(i*0.05)[1][0])
            mask_img_draw=ImageDraw.Draw(mask_img)
            mask_img_draw.polygon(pt_list,fill=(0,0,0))
            mask_tensor=get_tensor(normalize=False, toTensor=True)(mask_img)[0].unsqueeze(0)
        else:
            mask_img=np.zeros((H,W))
            mask_img[extended_bbox[1]:extended_bbox[3],extended_bbox[0]:extended_bbox[2]]=1
            mask_img=Image.fromarray(mask_img)
            mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)

        ### Crop square image
        if W > H:
            left_most=extended_bbox[2]-H
            if left_most <0:
                left_most=0
            right_most=extended_bbox[0]+H
            if right_most > W:
                right_most=W
            right_most=right_most-H
            if right_most<= left_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                left_pos=random.randint(left_most,right_most) 
                free_space=min(extended_bbox[1]-0,extended_bbox[0]-left_pos,left_pos+H-extended_bbox[2],H-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
                mask_tensor_cropped=mask_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
        
        elif  W < H:
            upper_most=extended_bbox[3]-W
            if upper_most <0:
                upper_most=0
            lower_most=extended_bbox[1]+W
            if lower_most > H:
                lower_most=H
            lower_most=lower_most-W
            if lower_most<=upper_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                upper_pos=random.randint(upper_most,lower_most) 
                free_space=min(extended_bbox[1]-upper_pos,extended_bbox[0]-0,W-extended_bbox[2],upper_pos+W-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
                mask_tensor_cropped=mask_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
        else:
            image_tensor_cropped=image_tensor
            mask_tensor_cropped=mask_tensor

        
        image_tensor_resize=self.resize(image_tensor_cropped)
        mask_tensor_resize=self.resize(mask_tensor_cropped)
        
        if self.state == "train":
            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            image_tensor_resize=self.augmentation(image_tensor_resize)
            
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            mask_tensor_resize=self.augmentation(mask_tensor_resize)
            
            
        inpaint_tensor_resize=image_tensor_resize*mask_tensor_resize
 
        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize,"ref_imgs":ref_image_tensor,"real_mask":1 - mask_tensor_resize}



    def __len__(self):
        return self.length



