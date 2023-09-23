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

`unzip diabetic-retinopathy-detection.zip "train*"` (~25 min)

`rm diabetic-retinopathy-detection.zip ` # Do this **after** running the unzip command.

`cat train.zip.* > train.zip` (~20 min)

`unzip train.zip` (~15 min)

`unzip trainLabels.csv.zip` (~1 min)

You'll now have a directory called `train/` with the images we'll use for this notebook.

To conduct DR Referral, the dataset needs to be binarized. Therefore, the data will be bucketed based on severity level into cateogries of 'no refer' and 'refer.' Bucketing occurred as follows:
- [No DR, mild] are 'no refer'
- [Moderate, Severe, Proliferate] are 'refer'

## Training

A Resnet-50 model, pre-trained on ImageNet, was used for classification. Various experiements were performed at varying model configurations, learning rates, and batch sizes. These experiments are summarized below:

| Experiment # | Model Configuration | Learning Rate | Batch Size |
| :---:        |       :----:        |  :---: | :---: |
| Experiment 1 | Layers 1-3 are frozen | 1e-5   | 64 |
| Experiment 2 | Layers 1-3 are frozen | 1e-4   | 64 |
| Experiment 3 | Layers 1-4 are frozen | 1e-4   | 128 |
| Experiment 4 | Layers 1-4 are frozen | 1e-3   | 128 |
| Experiment 5 | Layers 1-4 are frozen | 1e-3   | 64 |

## Results

| Experiment # | Accuracy | Precision | Recall | ROC |
|   :---:|  :---:|  :---:|  :---:|   :---:|
|   1|  0.6763|  0.7044| 0.6265|   0.6771|
|   2|  0.7025|  0.6856| 0.7696|   0.7011|
|   3|  0.6963|  0.7227| 0.6610|   0.6972|
|   4|  0.6937|  0.6921| 0.6580|   0.6925|
|   5|  0.7025|  0.6898| 0.6789|   0.7014|

