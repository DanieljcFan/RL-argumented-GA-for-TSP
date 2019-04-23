import pandas as pd
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
os.chdir('/Users/grapeeffect/Documents/deep learning')

g4 = pd.read_csv('graphs6.txt',skiprows=8,header = None)
g4 = g4.iloc[:,0].str.split(" ", n = 11, expand = True) 
g4
y=g4.iloc[:,-1].eq('Y').mul(1).values
y=[[y[i]] for i in range(len(y))]
X=g4.iloc[:,0:-1].astype(int).values
X.shape
y.shape


class My_Neural_Net(nn.Module):    
    #constructor
    def __init__(self):
        super(My_Neural_Net, self).__init__()
        
        # Define the layers. This matches the image above 
        # Except that our input size is 1000 dimensions
        self.layer_1 = nn.Linear(6, 4)
        self.layer_2 = nn.Linear(4,4)
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
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
        X = self.sigmoid(X)
        
        # 8. output predictions
        return X
    
    def loss(self, pred, true):
        #PyTorch's own cross entropy loss function.
        score = nn.BCELoss()
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
    
    def predict_proba(self, X):
        # probability of being a 1
        prob_1 = self.forward(X)
              
        # vectorwise subtraction
        prob_0 = 1 - prob_1
        
        # make into a matrix
        probs = torch.cat((prob_0,prob_1), dim = 1)
        
        return probs
    
    def predict(self, X):
        probs = self.predict_proba(X)
        
        # get only second column (probability of being a 1)
        probs_1 = probs[:,1:]
        
        # 1 if prob_1 is greater or equal to than 0.5 for a given example
        # 0 if less than 0.5
        preds = (probs_1 >= 0.5).int()
        
        return preds
    
    def score(self, X, y):
        # proportion of times where we're correct
        # conversions just allow the math to work
        acc = (self.predict(X) == y.int()).float().mean()
        
        return acc
    

#Create our neural net
neur_net = My_Neural_Net()

# Split into train and test so we can fit on some data and see performance 
# on data we havent seen yet.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Turn X and y (train and test) into PyTorch objects. We always have to do this step
X_train_tens = Variable(torch.Tensor(X_train).float())
X_test_tens = Variable(torch.Tensor(X_test).float())
y_train_tens = Variable(torch.Tensor(y_train).float())
y_test_tens = Variable(torch.Tensor(y_test).float())


neur_net.fit(X_train_tens,y_train_tens)
neur_net.score(X_train_tens,y_train_tens)
neur_net.score(X_test_tens,y_test_tens)
