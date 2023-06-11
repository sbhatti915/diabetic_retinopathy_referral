# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

path_to_csv = '/home/sameer/biods220/assign1/diabetic_retinopathy_referral/diabetic_retinopathy/trainLabels.csv'

df = pd.read_csv(path_to_csv)

# plt.hist(df['level'], bins=5, rwidth=0.6)

# fig, ax = plt.subplots()
x = df['level'].value_counts()
levels = list(x.index.values)
counts = x.tolist()
# plxt(ax=ax, kind='bar')

plt.bar(levels, counts)
plt.title('Counts of levels')
plt.xlabel('Levels')
plt.ylabel('Frequency')