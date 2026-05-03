# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:12:03 2026

@author: Jeco
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Subset

from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

from pathlib import Path
# example from: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

#%% Paths
script_dir = Path(__file__).resolve().parent

#dataLoc=script_dir.parent / r'sorted_pytorch\zzz_outputs_m120dBFloor_sitPulseRef_Centered_RowFFT'
dataLoc= script_dir.parent / r'sorted_pytorch\zzz_outputs_m120dBFloor_sitPulseRef_Centered_ColFFT'

dataLoc=str(dataLoc)
    
#%% Plotting functions
def plot_accuracy_curves(train_acc, val_acc):
    # tempTrn= cp.asnumpy(cp.array(train_acc))
    # tempVal= cp.asnumpy(cp.array(val_acc))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Plots')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def plot_learning_curves(train_losses, val_losses):
    # tempTrn= cp.asnumpy(cp.array(train_losses))
    # tempVal= cp.asnumpy(cp.array(val_losses))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.grid(True)
    plt.legend()
    plt.show()

#%% Confusion Matrix
def create_confusion_matrix(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs= inputs.to(device)
            labels= labels.to(device)
            outputs = model(inputs)
                
            preds = torch.argmax(outputs, axis=1)
            all_preds.extend(preds.tolist())
            
            if labels.ndim == 0: # scalar label
                all_labels.append(labels.item())
            else:
                all_labels.extend(labels.cpu().numpy().flatten())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm_normalized


#%% Check for nvidia
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%% CNN Class
class CNN0(nn.Module):
    def __init__(self):
        super(CNN0, self).__init__()
        self.conv1= nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu= nn.ReLU()
        self.pool= nn.MaxPool2d(2)
        self.conv2= nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.flatten= nn.Flatten()
        self.fc= nn.Linear(32 * 85 * 108, 11) # for input of 3, 343, 434
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Supports
        self.relu= nn.ReLU()
        self.pool= nn.MaxPool2d(2)
        self.flatten= nn.Flatten()
        self.dropout= nn.Dropout(0.2)#on in training, off in eval
        self.avgpool= nn.AdaptiveAvgPool2d((1,1))
        
        # Block1
        self.conv1= nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.batch1= nn.BatchNorm2d(16)
        
        #Block2
        self.conv2= nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.batch2= nn.BatchNorm2d(32)#must match dimension of output conv
        
        #Block3
        self.conv3= nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batch3= nn.BatchNorm2d(64)
        
        #Hidden before Output
        self.hl= nn.Linear(64 * 42 * 54, 256)
        
        #Output Layer
        self.ol= nn.Linear(64 * 42 * 54, 11) # for input of 3, 343, 434
        self.ol2= nn.Linear(256, 11) # use for experimental with 2nd layer
        
    def forwardExp(self, x):# Experiment
        x= self.pool(self.relu(self.batch1(self.conv1(x))))# a messy instance of a block
        x= self.pool(self.relu(self.batch2(self.conv2(x))))
        x= self.pool(self.relu(self.batch3(self.conv3(x))))
        #x= self.flatten(self.avgpool(x))
        x= self.flatten(x)
        x= self.relu(self.hl(x))
        x= self.dropout(x)
        x= self.ol2(x)
        return x
    
    def forward(self, x):# Simple
        x= self.pool(self.relu(self.batch1(self.conv1(x))))# a messy instance of a block
        x= self.pool(self.relu(self.batch2(self.conv2(x))))
        x= self.pool(self.relu(self.batch3(self.conv3(x))))
        x= self.flatten(x)
        x= self.dropout(x)
        x= self.ol(x)
        return x



#%% Main
if __name__ == '__main__':
    
    #%% Data Load
    classNames=["walking_towards", "walking_away", "picking_obj", "Bending", 
                "sitting", "kneeling", "crawling", "walking_on_toe", "limping_RL",
                "short_steps", "scissor_gait"]

    transform = transforms.Compose([transforms.ToTensor()])
    
    batchSz=8

    # Load Train
    train_dataset = datasets.ImageFolder(root= dataLoc+r'\train', transform= transform)
    #train_loader= DataLoader(train_dataset, batch_size=batchSz, shuffle=True)# keep true to ensure random training
    
    # Split Train into train and validation, ensure that classes are still proportional
    train_idx, validation_idx = train_test_split(np.arange(len(train_dataset)),
                                             test_size=0.223,
                                             random_state=123,
                                             shuffle=True,
                                             stratify=train_dataset.targets)
    
    train_subset = Subset(train_dataset, train_idx)
    train_loader = DataLoader(train_subset, batch_size=batchSz, shuffle=True)
    
    validation_subset = Subset(train_dataset, validation_idx)
    validation_loader = DataLoader(validation_subset, batch_size=len(validation_subset), shuffle=False)
    
    # Load Test
    test_dataset = datasets.ImageFolder(root= dataLoc+r'\test', transform= transform)
    test_loader= DataLoader(test_dataset, batch_size=batchSz, shuffle=False)# keep false for consistent diagnostics
    
    images, labels = next(iter(train_loader))#sample image
    
    #%% Setup
    
    model= CNN().to(device)
    model.train()#set to train to enable neuron dropout
    
    criterion= nn.CrossEntropyLoss()
    #optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer= optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optimizer= optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)
    epochs=20# keep to 50, high number of epochs
    
    # Fully loaded valiation
    inputsVal, labelsVal = next(iter(validation_loader))
    inputsVal= inputsVal.to(device)
    labelsVal= labelsVal.to(device)
    
    # Storage for Recordings
    accListTrn=[]
    accListVal=[]
    lossListTrn=[]
    lossListVal=[]
    
    #%% Train
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        startEpoch=True
        for batch, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs=inputs.to(device)
            labels= labels.to(device)
            
            optimizer.zero_grad()# reset parameters
                  
            outputs= model(inputs)# forward
            loss= criterion(outputs, labels)
            loss.backward()# backward
            optimizer.step()# optimize
            
            # print statistics
            running_loss += loss.item()
            if batch % batchSz == batchSz-1:    # print every 5 mini-batches
                print(f'[{epoch + 1}, {batch + 1:5d}] loss: {running_loss / batchSz:.5f}')
                running_loss = 0.0
                
            if startEpoch:
                # Run validation
                outputs_val= model(inputsVal)
                loss_val= criterion(outputs_val, labelsVal)
                
                # Compute Accuracy 
                accListTrn.append(int((outputs.max(1)[1]==labels).sum())/len(outputs))
                accListVal.append(int((outputs_val.max(1)[1]==labelsVal).sum())/len(outputs_val))
                
                # Record Metrics
                lossListTrn.append(float(loss))
                lossListVal.append(float(loss_val))
                
                startEpoch=False
    
    print('Finished Training')
    
    #%% Check Test
    model.eval()
    accList=[]
    
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs=inputs.to(device)
        labels= labels.to(device)
              
        outputs= model(inputs)# forward
        accList.append(int((outputs.max(1)[1]==labels).sum()))
        
    print(f'Accurracy is about:{sum(accList)/len(test_loader.dataset)}')
    
    #%% Plotting
    plot_accuracy_curves(accListTrn, accListVal)
    plot_learning_curves(lossListTrn, lossListVal)
    
    
    #%% Confusion Matrix
    cm= create_confusion_matrix(model=model, data_loader=test_loader, device=device)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=True, xticklabels=classNames, yticklabels=classNames)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
#%% Notes
"""Sources
## Train vs Eval modes
- https://yassin01.medium.com/understanding-the-difference-between-model-eval-and-model-train-in-pytorch-48e3002ee0a2

## Stratified Split
- https://medium.com/@heyamit10/how-to-split-a-custom-dataset-into-training-and-test-datasets-a-practical-guide-54d5043c67e3
- https://discuss.pytorch.org/t/how-to-do-a-stratified-split/62290

## Optimization
- https://gurjeet333.medium.com/7-best-techniques-to-improve-the-accuracy-of-cnn-w-o-overfitting-6db06467182f
- https://towardsdatascience.com/exploring-the-superhero-role-of-2d-batch-normalization-in-deep-learning-architectures-b4eb869e8b60/
- https://arxiv.org/abs/1502.03167
"""

""" Results for ColFFT (downwards is chrono-order)
## 57.25% accurracy
- 50 epochs with lr=0.001, momentum=0.9, batchsize=16

## 59.54% accurracy
- 100 epochs with lr=0.001, momentum=0.9, batchsize=16

## 64.88% accuracy
- 50 epochs, optimizer adam, lr=0.0001, default betas,  batchsize=16 

## 68.7% acc
- 100 epochs, optimizer adam, lr=0.0001, default betas, batchsize=8

## 65.64% acc
- 50 epochs, optimizer adam, lr=0.0001, default betas, batchsize=8, added dropout

## 67.9% acc
- 50 epochs, optimizer adam, lr=0.0001, default betas, batchsize=8, added batch norm
"""


""" Results for RowFFT
## 54.96% acc
- 50 epochs, optimizer adam, lr=0.0001, default betas, batchsize=4, with dropout and batch norm

## 62.59% acc
- 50 epochs, ..., changed dropout rate from 0.5 to 0.2

"""
