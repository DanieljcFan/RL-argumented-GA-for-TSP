#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 23:12:15 2019

@author: daniel
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

class My_Neural_Net(nn.Module): 
    
    #constructor
    #take in X as a parameter
    def __init__(self, X):
        super(My_Neural_Net, self).__init__()
        
        #Find dimensionality of X
        X_dim = X.shape[1]
        
        # Define the layers. This matches the image above 
        # Except that our input size is X_dim dimensions
        self.layer_1 = nn.Linear(X_dim, 10)
        self.layer_2 = nn.Linear(10,4)
        self.layer_3 = nn.Linear(4,1)

        # Define activation functions. I'll be using ReLU
        # for the hidden layers. Must use sigmoid for the 
        # final layer so we get a number between 0 and 1 for 
        # the probability of being about baseball.
        # Luckily PyTorch already has ReLU and 
        # sigmoid.
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Define what optimization we want to use.
        # Adam is a popular method so I'll use it.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.5)
        
    # 1. input X
    def forward(self, X):
        # 2. linearly transform X into hidden data 1
        X = self.layer_1(X)
        # 3. perform ReLU on hidden data
        X = self.relu(X)
        # 4. linearly transform hidden data into hidden data 2
        X = self.layer_2(X)
        # 5. perform ReLU on hidden data
        X = self.relu(X)
        # 6. linearly transform hidden data into output layer
        X = self.layer_3(X)
        # 7. perform sigmoid on output data to get f(X) predictions between 0 and 1
        #X = self.sigmoid(X)
        
        # 8. output predictions
        return X
    
    def loss(self, pred, true):
        #PyTorch's own cross entropy loss function.
        score = nn.MSELoss()
        return score(pred, true)
    

    # 1. input: N - number of iterations to train, X - data, y - target
    def fit(self,X,y,N = 5000):
        
        # 2. for n going from 0 to N -1 :
        for epoch in range(N):
            
            # Reset weights in case they are set for some reason
            self.optimizer.zero_grad()
            
            # 3. f(X) = forward(X) 
            pred = self.forward(X)
            
            # 4. l = loss(f(X),y)
            l = self.loss(pred, y)
            #print loss
            print(l)
            
            # 5. Back progation
            l.backward()
            # 5. Gradient Descent
            self.optimizer.step()
    
    def predict(self, X):
        preds = self.forward(X)                
        return preds
    
    def score(self, X, y):
        # proportion of times where we're correct
        # conversions just allow the math to work
        mse = ((self.predict(X) - y)**2).mean()
        
        return mse


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size()) + 200                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

mynn = My_Neural_Net(x)
mynn.fit(x,y)
mynn.predict(x)

plt.figure()
plt.scatter(x.data.numpy(), y.data.numpy(), color = "orange")
plt.scatter(x.data.numpy(), mynn.predict(x).data.numpy(), color = "blue")








