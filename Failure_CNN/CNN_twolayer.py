#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:16:56 2019

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
import random
import os
os.chdir('/Users/daniel/Documents/academic/DeepLearning/NN-agumented-GA-for-TSP')

from route import City, readCities
cts = readCities('gr96_tsp.txt',7)
maps = [City(c[0],c[1],i) for i,c in enumerate(cts)]


leaveone = pd.read_csv("leave1_1step.csv")
y = leaveone.values[:,-1]
#y = [1 if i>0 else 0 for i in y]
routes = leaveone.values[:,:96]
n = len(routes[0])
D = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        d = maps[i].distance(maps[j])
        D[i,j] = d
        D[j,i] = d

D_c = [np.mean(D),np.std(D)]
D = (D-D_c[0])/D_c[1]


res = []
for route in routes:
    n = len(route)
    X = np.zeros((n,n))
    for i in range(1,n):    
        d = maps[int(route[i-1])].distance(maps[int(route[i])])
        X[int(route[i-1]),int(route[i])] = d
        X[int(route[i]),int(route[i-1])] = d
    
    #res.append([X,D])
    res.append(X)
X = np.array(res)[:,None,:,:]


#How to combine image and label together??
batch_size = 100
Y = Variable(torch.Tensor(y).float())
#Y = Variable(y)
data_all = []
for i in range(len(X)):
    data_all.append([X[i],Y[i]])

ratio_train = 0.2
cut = int(len(data_all)*ratio_train)
data_train = data_all[:cut]
data_test = data_all[cut:]
X_train = X[:cut]
X_test = X[cut:]
y_train = y[:cut]
y_test = y[cut:]


train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*8*8, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        #self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print("begin:", x.size())
        x = x.float()
        #print("begin:", x.size())
        x = self.bn1(self.pool(F.relu(self.conv1(x))))
        #print("conv1:", x.size())
        x = self.bn2(self.pool(F.relu(self.conv2(x))))
        #print("conv2:", x.size())
        x = self.bn3(self.pool(F.relu(self.conv3(x))))
        #print("conv3:", x.size())
        x = x.view(-1, 64*8*8)
        #print("view:", x.size())
        x = F.relu(self.fc1(x))
        #print("fc1:", x.size())
        x = F.relu(self.fc2(x))
        #print("fc2:", x.size())
        x = self.fc3(x)
        #print("fc3:", x.size())
        #x = self.Sigmoid(x)
        return x


net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)
running_loss = 0.0

print('Training begins')
for j in range(5):
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
        running_loss = loss.item()
        #if i % 2000 == 2000-1:#print every 2000 mini-batches
    print('[%5d] loss: %.3f' %
          (j, running_loss))
    running_loss = 0.0
print("Finished Training")

y_fit = []
with torch.no_grad():
    for data in train_loader:
        inputs, score  = data
        inputs = inputs.float()
        y = net(torch.Tensor(inputs).float())
        #print(criterion(y,score).item())
        y_fit = y_fit + [i.detach().numpy() for i in y]


rank = sorted(range(len(y_train)), key=lambda i: y_train[i])
y_fit = np.array(y_fit)


1 - (np.std(y_fit-y_train)/np.std(y_train))**2

plt.figure()
plt.plot(y_fit[rank])
#plt.figure()
plt.plot(y_train[rank])



y_pred = []
with torch.no_grad():
    for data in test_loader:
        inputs, score  = data
        inputs = inputs.float()
        y = net(torch.Tensor(inputs).float())
        criterion(y,score).item()
        y_pred = y_pred + [i.detach().numpy() for i in y]
y_pred = np.array(y_pred)

1- (np.std(y_test-y_pred)/np.std(y_test))**2


rank = sorted(range(len(y_test)), key=lambda i: y_test[i])

plt.figure()
plt.plot(y_pred[rank])
#plt.figure()
plt.plot(y_test[rank])


def Rankloss(pred,true,k=1000):
    error = 0
    n = len(pred)
    for _ in range(k):
        i,j = random.sample(range(n),2)
        if (pred[i] - pred[j])*(true[i]-true[j])<0:
            error +=1
    return error/k

Rankloss(y_fit, y_train)
Rankloss(y_pred,y_test)





