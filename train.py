#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 23:14:54 2023

@author: sameer
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import DRR
import pandas as pd
import matplotlib.pyplot as plt

root_path = '/home/sameer/biods220/assign1/diabetic_retinopathy_referral/diabetic_retinopathy/train'
csv_path = '/home/sameer/biods220/assign1/diabetic_retinopathy_referral/diabetic_retinopathy/trainLabels.csv'
batch_size = 64
num_epochs = 25
learning_rate=0.00001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
combined_df = combined_df.sample(frac = 1)

dataset = DRR(combined_df, root_path, transform)

train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

sample = next(iter(train_loader))

example = transforms.ToPILImage()(sample[0][0].squeeze()) # sample[0][0] for 1st image in batch, sample[0] for batch of 1

plt.imshow(example)

# Load a pre-trained ResNet model
print("Loading Model")
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

# Freeze last and second to last layers
for name, param in model.named_parameters():
    if 'layer4' or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
    
# Modify the last layer for binary classification (assuming the number of features in the last layer is 2048 for ResNet-50)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 1),  # 1 output unit for binary classification
    nn.Sigmoid()  # Sigmoid activation for binary classification
)

model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_f1_val = 0

print("Beginning Training")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    predictions = []
    ground_truth = []

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Convert predictions to binary values (0 or 1) using a threshold (e.g., 0.5 for binary classification)
        predictions.extend((outputs > 0.5).squeeze().cpu().tolist())  # Assuming binary classification
        # Store ground truth labels
        ground_truth.extend(targets.cpu().tolist())
        
        loss = criterion(outputs, targets.unsqueeze(1).float()) # Have to unsqueeze to match shape for loss function
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    f1_train = f1_score(ground_truth, predictions)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, f1 score: {f1_train}")

    # Validation
    model.eval()
    predictions = []
    ground_truth = []
    total_val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            # Convert predictions to binary values (0 or 1) using a threshold (e.g., 0.5 for binary classification)
            predictions.extend((outputs > 0.5).squeeze().cpu().tolist())  # Assuming binary classification
            # Store ground truth labels
            ground_truth.extend(targets.cpu().tolist())
            val_loss = criterion(outputs, targets.unsqueeze(1).float()) 

            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    f1_val = f1_score(ground_truth, predictions)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, f1 score: {f1_val}")
    
    if f1_val > best_f1_val:
        best_f1_val = f1_val
        torch.save(model.state_dict(), '/home/sameer/biods220/assign1/diabetic_retinopathy_referral/model.pt')
        print('Model saved')
    print("Finished Epoch")

print("Training complete. Starting Evaluation. Loading trained model")
# Evaluate
model.load_state_dict(torch.load('/home/sameer/biods220/assign1/diabetic_retinopathy_referral/model.pt'))
print("Trained model loaded")
model.eval()  # Switch to evaluation mode

# Create an empty list to store the predictions and ground truth labels
predictions = []
ground_truth = []

# Optional: use torch.no_grad() for memory efficiency during evaluation
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)  # Move inputs to the same device as the model (GPU/CPU)
        targets = targets.to(device)  # Move targets to the same device as the model (GPU/CPU)

        # Forward pass (calculate predictions)
        outputs = model(inputs)

        # Convert predictions to binary values (0 or 1) using a threshold (e.g., 0.5 for binary classification)
        predictions.extend((outputs > 0.5).squeeze().cpu().tolist())  # Assuming binary classification

        # Store ground truth labels
        ground_truth.extend(targets.cpu().tolist())

accuracy = accuracy_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions)
recall = recall_score(ground_truth, predictions)
f1 = f1_score(ground_truth, predictions)

print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
