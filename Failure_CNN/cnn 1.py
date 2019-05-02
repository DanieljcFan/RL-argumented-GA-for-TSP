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

from route import City, readCities
cts = readCities('gr96_tsp.txt',7)
maps = [City(c[0],c[1],i) for i,c in enumerate(cts)]


trainset = pd.read_csv("trainset_1000.csv")
y_train = trainset.iloc[:,-2]
routes = trainset.values[:,:96]
res = []
for route in routes:
    n = len(route)
    X = np.zeros((n,n))
    for i in range(n):    
        d = maps[int(route[i-1])].distance(maps[int(route[i])])
        X[int(route[i-1]),int(route[i])] = d
        X[int(route[i]),int(route[i-1])] = d
    
    res.append(X)
X_train = np.array(res)[:,None,:,:]

testset = pd.read_csv("trainset_100.csv")
y_test = testset.values[:,-2].reshape(-1,1)
routes = testset.values[:,:96]
res = []
for route in routes:
    n = len(route)
    X = np.zeros((n,n))
    for i in range(n):    
        d = maps[int(route[i-1])].distance(maps[int(route[i])])
        X[int(route[i-1]),int(route[i])] = d
        X[int(route[i]),int(route[i-1])] = d
    
    res.append(X)
X_test = np.array(res)[:,None,:,:]


#How to combine image and label together??
batch_size = 100
y_c = [np.mean(y_train), np.std(y_train)]
Y_train = Variable(torch.Tensor((y_train-y_c[0])/y_c[1]).float())
data_train = []
for i in range(len(X_train)):
    data_train.append([X_train[i],Y_train[i]])




train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(dataset=X_test, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*21*21, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)

    def forward(self, x):
        #print("begin:", x.size())
        x = x.float()
        #print("begin:", x.size())
        x = self.pool(F.relu(self.conv1(x)))
        #print("conv1:", x.size())
        x = self.pool(F.relu(self.conv2(x)))
        #print("conv2:", x.size())
        x = x.view(-1, 16 * 21* 21)
        #print("view:", x.size())
        x = F.relu(self.fc1(x))
        #print("fc1:", x.size())
        x = F.relu(self.fc2(x))
        #print("fc2:", x.size())
        x = self.fc3(x)
        #print("fc3:", x.size())
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
        running_loss += loss.item()
        #if i % 200 == 200-1:#print every 2000 mini-batches
    print('[%5d] loss: %.3f' %
          (j, running_loss / 2000))
    running_loss = 0.0
print("Finished Training")



y_pred = net(torch.Tensor(X_test).float())
y_pred = np.array([i.detach().numpy() for i in y_pred])

rank = sorted(range(len(y_test)), key=lambda i: y_test[i])

plt.figure()
plt.plot(y_pred[rank])

plt.figure()
plt.plot(y_test[rank])


