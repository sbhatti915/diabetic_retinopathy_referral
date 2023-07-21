#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 23:14:54 2023

@author: sameer
"""
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import DRR
import pandas as pd
import matplotlib.pyplot as plt

root_path = '/home/sameer/biods220/assign1/diabetic_retinopathy_referral/diabetic_retinopathy/train'
csv_path = '/home/sameer/biods220/assign1/diabetic_retinopathy_referral/diabetic_retinopathy/trainLabels.csv'
batch_size = 4

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

df = pd.read_csv(csv_path)
filtered_df = df[df['level'] >= 2]
sampled_df_refer = filtered_df.sample(n=2000, random_state=42)
filtered_df = df[df['level'] < 2]
sampled_df_no_refer = filtered_df.sample(n=2000, random_state=42)
combined_df = pd.concat([sampled_df_no_refer, sampled_df_refer], ignore_index=True)

x = DRR(combined_df, root_path, transform)
data_loader = DataLoader(x, batch_size=batch_size, shuffle=True)

sample = next(iter(data_loader))

example = transforms.ToPILImage()(sample[0][0].squeeze())

plt.imshow(example)