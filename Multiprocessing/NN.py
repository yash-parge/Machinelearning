#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from multiprocessing import Process, Pool, Lock
import numpy as np
import pandas as pd
import random
data = pd.read_csv(
    "file_path",
    usecols=['1', '2', '3'],
    dtype={"1": float, "2": float, "3": float}
)
data['Bias'] = 0.0
data = data.to_numpy()
#data = data/np.amax(data, axis=0)
#imports the answers to the dataset which we will set as the closing price
print(data.shape)
data_ans = pd.read_csv(
    "file_path",
    usecols=['Result'],
    dtype={"Result": float}
)

data_ans = data_ans.to_numpy()
print(data_ans.shape)
#data_ans = data_ans/np.amax(data_ans, axis=0)


# In[2]:


class NeuralNetwork:
    def __init__(self, seed):
        self.inputsize = 3 + 1 #Number of inputs given + bias node
        self.hiddensize = 5#Number of nodes in the hidden layer
        self.outputsize = 1 #Number of nodes in the output layer
        np.random.seed(seed)
        self.bias_2 = np.random.randn(self.hiddensize)
        self.W1 = np.random.randn(self.inputsize, self.hiddensize)
        self.W2 = np.random.randn(self.hiddensize, self.outputsize)
        
    def forwardprop(self, input):
        self.input = input
        self.z1 = np.dot(input, self.W1) #Dot function of inputs and out first weights
        self.z1 = self.z1 + self.bias_2
        self.z1activ = self.sigmoid(self.z1) #Activation of hidden layer outputs
        self.z2 = np.dot(self.z1activ, self.W2) #Dot function of hidden layer and our second weights
        #print(self.z2.shape)
        self.out = self.sigmoid(self.z2)
        return self.out
    
    #Sigmoid and sigmoid derivitive functions
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1/(1+np.exp(-x))

    
    def backwardprop(self, input, answers):
        self.outputerror = answers - self.out #This will be our loss function
        #print(self.outputerror)
        egg = ((0.07 * self.outputerror) * self.sigmoid(self.out, True))
        weight1 = np.dot(self.input.T, (np.dot(egg, self.W2.T)) * self.sigmoid(self.z1activ, True))
        weight2 = np.dot(self.z1activ.T, egg)
        self.W1 += weight1 #updating of weights
        self.W2 += weight2
        self.bias_2 += (0.05 * np.sum(weight1, axis=0)) #updating of bias
        
    def train (self, dataset, answers, repetition):
        for i in range(repetition):
            self.forwardprop(dataset) #forwardpropagation
            self.backwardprop(dataset, answers)


# In[3]:

def NN(num, list):
    NQ = NeuralNetwork(num)
    m = NQ.W2
    NQ.train(data, data_ans, 10)
    l = NQ.forwardprop(np.array([0.141421356, 0.424264069, 4.0, 1.0]))
    q = [l, num]
    list.put(q) #use of queue to transer variables across files


# In[6]:





# In[ ]:




