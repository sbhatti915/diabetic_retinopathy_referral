# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
import os
import PIL

path_to_csv = '/home/sameer/biods220/assign1/diabetic_retinopathy_referral/diabetic_retinopathy/trainLabels.csv'
path_to_images = '/home/sameer/biods220/assign1/diabetic_retinopathy_referral/diabetic_retinopathy/train/'

def load_csv(csv_path):
    dataframe = pd.read_csv(path_to_csv)
    return dataframe

def view_counts(dataframe):
    
    x = dataframe['level'].value_counts()
    levels = list(x.index.values)
    counts = x.tolist()

    plt.bar(levels, counts)
    plt.title('Counts of levels')
    plt.xlabel('Levels')
    plt.ylabel('Frequency')
    return counts

def view_images(dataframe, images_path, num_levels):
    for k in range(num_levels):
        plt.figure()
        level_dataframe = dataframe.loc[dataframe['level'] == k]
        image_name = random.choice(list(level_dataframe['image']))
        path_to_image = os.path.join(images_path, f'{image_name}.jpeg')
        image = PIL.Image.open(path_to_image)
        plt.title(f"Severity = {str(k)}")
        plt.imshow(image)
        
if __name__ == '__main__':
    df = load_csv(path_to_csv)
    counts = view_counts(df)
    view_images(df, path_to_images, len(counts))
    