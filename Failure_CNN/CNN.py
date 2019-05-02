#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:26:07 2019

@author: daniel
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import os
os.chdir('/Users/daniel/Documents/academic/DeepLearning/NN-agumented-GA-for-TSP')
trainset = pd.read_csv("trainset_1000.csv")
y_train = trainset.iloc[:,-3]
routes = trainset.values[:,:96]
res = []
for route in routes:
    n = len(route)
    X = np.zeros((n,n))
    for i in range(n):    
        X[int(route[i-1]),int(route[i])] = 1
        X[int(route[i]),int(route[i-1])] = 1
    
    res.append(X)
X_train = np.array(res)[:,None,:,:]

testset = pd.read_csv("trainset_100.csv")
y_test = testset.iloc[:,-3]
routes = testset.values[:,:96]
res = []
for route in routes:
    n = len(route)
    X = np.zeros((n,n))
    for i in range(n):    
        X[int(route[i-1]),int(route[i])] = 1
        X[int(route[i]),int(route[i-1])] = 1
    
    res.append(X)
X_test = np.array(res)[:,None,:,:]


#How to combine image and label together??
batch_size = 100
y_train = Variable(torch.Tensor(y_train).float())
data_train = []
for i in range(len(X_train)):
    data_train.append([X_train[i],y_train[i] ])




train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(dataset=X_test, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)

    def forward(self, x):
        print("begin:", len(x))
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        print("conv1:", len(x))
        x = self.pool(F.relu(self.conv2(x)))
        print("conv2:", len(x))
        x = x.view(-1, 16 * 5 * 5)
        print("view:", len(x))
        x = F.relu(self.fc1(x))
        print("fc1:", len(x))
        x = F.relu(self.fc2(x))
        print("fc2:", len(x))
        x = self.fc3(x)
        print("fc3:",len(x))
        return x


net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
running_loss = 0.0

y_train = Variable(torch.Tensor(y_train).float())
#y_test = Variable(torch.Tensor(y_test).float())
for j in range(10):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, score = data
        score = score.float()
    
        # zero the parameter gradients
        optimizer.zero_grad()
    
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, score)
        loss.backward()
        optimizer.step()
    
        # print statistics
        #running_loss += loss.item()
        #if i < 200:    # print every 2000 mini-batches
        #    print('[%5d] loss: %.3f' %
        #          (i + 1, running_loss / 2000))
        #    running_loss = 0.0
print("Finished Training")

out_list = net(torch.Tensor(X_test).float())
out_list = [ i.detach().numpy() for i in out_list]

y_pred = (out_list - np.mean(out_list))/np.std(out_list)
y_test_c = (y_test)
