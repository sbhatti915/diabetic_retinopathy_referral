#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sameer

Dataset class for the Diabetic Retinopathy Referral. This takes in a dataframe
to obtain image file names and severity scores. This loads the image,
preprocesses it, and converts the corresponding severity score to a 0
(don't refer) or a 1 (refer).
"""

from torch.utils.data import Dataset
from PIL import Image
import os

class DRR(Dataset):
    def __init__(self, df, root_path, transforms=None):

        self.df = df
        self.root_path = root_path
        self.transforms = transforms
 
    def __len__(self):
        # this should return the size of the dataset
        return len(self.df)
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        img_file_name = self.df['image'][idx]
        level = self.df['level'][idx]
        # print(img_file_name)
        
        if (level >= 2):
            target = 1
        elif (level < 2):
            target = 0
        
        img_path = os.path.join(self.root_path, img_file_name)
        img = Image.open(f'{img_path}.jpeg')
        
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, target
        