#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:15:24 2019

@author: daniel
"""

import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import os

os.chdir('/Users/daniel/Documents/academic/DeepLearning/NN-agumented-GA-for-TSP')
trainset = pd.read_csv("trainset_100.csv")
score = trainset.iloc[:,-3]
y = np.array(score).reshape(score.shape[0],1)

routes = trainset.values[:,:96]
res = []
for route in routes:
    n = len(route)
    X = np.zeros((n,n))
    for i in range(n):    
        X[int(route[i-1]),int(route[i])] = 1
        X[int(route[i]),int(route[i-1])] = 1
    
    x = []
    for i in range(n):
        x = x+list(X[i][i+1:])
    res.append(x)

X = np.array(res)

class My_Neural_Net(nn.Module): 
    def __init__(self, X):
        super(My_Neural_Net, self).__init__()
        X_dim = X.shape[1]
        self.layer_1 = nn.Linear(X_dim,int(np.sqrt(X_dim)))
        self.layer_2 = nn.Linear(int(np.sqrt(X_dim)),int(np.log(X_dim)))
        self.layer_3 = nn.Linear(int(np.log(X_dim)),1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1) ##### consider about learning rate
    def forward(self, X):
        X = self.layer_1(X)
        X = self.relu(X)
        X = self.layer_2(X)
        X = self.relu(X)
        X = self.layer_3(X)
        X = self.relu(X)
        return X
    def loss(self, pred, true):
        score = nn.MSELoss()
        return score(pred, true)
    def fit(self,X,y,N = 5000):
        for epoch in range(N):
            self.optimizer.zero_grad()
            pred = self.forward(X)
            l = self.loss(pred, y)
            #print(l)
            l.backward()
            self.optimizer.step()
    def predict(self, X):
        output = self.forward(X)
        return output
    def score(self, X, y):
        diff = (self.predict(X)-y)
        acc = diff**2
        return diff

X = Variable(torch.Tensor(X).float())
y = Variable(torch.Tensor(y).float())
neur_net = My_Neural_Net(X)

neur_net.fit(X,y)

neur_net.predict(X)




