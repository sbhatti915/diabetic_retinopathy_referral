# Diabetic Retinopathy Referral

## Introduction

Diabetic retinopathy (DR) is an eye disease prevalent in diabetic patients. It is the leading cause of blindness in people aged 20-64. Screening for DR allows earlier and more effective treatment options, and accurate screening can save the eyesight of millions. A deep-learning approach to predicting DR from eye images was proposed in 2016, in the paper "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs", published in JAMA.

In this code base, a deep learning model will be trained on fundus photos to predict if a patient should be referred for treatment given binarized severity of DR in patients: no referral if [No DR, mild] and referral if [moderate, severe, and proliferate DR].

This project is a modified assignment from Stanford's BIODS 220 course taught by Dr. Serena Yeung. Course information can be found here: http://biods220.stanford.edu/

## Dataset 

The Diabetic Retinopathy Detection dataset was used for this project which is hosted on Kaggle. The liink to the dataset is here: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data.

The dataset is of retina images from both left and right eyes. it also contains a csv file with the image names and the corresponding severity levels of DR. A clinician has rated the presence of diabetic retinopathy in each image on a scale of 0 to 4, according to the following scale:

0 - No DR

1 - Mild

2 - Moderate

3 - Severe

4 - Proliferative DR

To use the dataset for training, download the files from kaggle. Then run the following commands:

`unzip diabetic-retinopathy-detection.zip "train*` (~25 min)

`rm diabetic-retinopathy-detection.zip ` # Do this **after** running the unzip command.

`cat train.zip.* > train.zip` (~20 min)

`unzip train.zip` (~15 min)

`unzip trainLabels.csv.zip` (~1 min)

You'll now have a directory called `train/` with the images we'll use for this notebook.

To conduct DR Referral, the dataset needs to be binarized. Therefore, the data will be bucketed based on severity level into cateogries of 'no refer' and 'refer.' To do so, bucketing occurred as  follows:
-  [No DR, mild] are 'no refer'
- [Moderate, Severe, Proliferate] are 'refer'

