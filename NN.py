#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
#imports dataset
data = pd.read_csv(
    "Filepath to training dataset",
    usecols=['1', '2', '3'],
    dtype={"1": float, "2": float, "3": float}
)
data['Bias'] = 0.0 # Adds bias coloumn to dataset
data = data.to_numpy()
#imports the results coloumn of the dataset
data_ans = pd.read_csv(
    "Filepath to training dataset",
    usecols=['Result'],
    dtype={"Result": float}
)

data_ans = data_ans.to_numpy()


# In[2]:


class NeuralNetwork(object):
    def __init__(self):
        self.inputsize = 3 + 1 #Number of inputs given + bias node
        self.hiddensize = 5#Number of nodes in the hidden layer
        self.outputsize = 1 #Number of nodes in the output layer
        np.random.seed(1)
        self.bias_2 = np.random.randn(self.hiddensize)
        self.W1 = np.random.randn(self.inputsize, self.hiddensize)
        self.W2 = np.random.randn(self.hiddensize, self.outputsize)
        
    def forwardprop(self, input):
        self.input = input
        self.z1 = np.dot(input, self.W1) #Dot function of inputs and out first weights
        self.z1 = self.z1 + self.bias_2 # Adds bias
        self.z1activ = self.sigmoid(self.z1) #Activation of hidden layer outputs
        self.z2 = np.dot(self.z1activ, self.W2) #Dot function of hidden layer and our second weights
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
        weight1 = np.dot(self.input.T, (np.dot(egg, self.W2.T)) * self.sigmoid(self.z1activ, True)) # Calculates error in weight 1
        weight2 = np.dot(self.z1activ.T, egg) # Calculates error in weight 2
        self.W1 += weight1 # updates weight 1
        self.W2 += weight2 # updates weight 2
        self.bias_2 += (0.05 * np.sum(weight1, axis=0)) # updates bias
        
    def train (self, dataset, answers, repetition):
        for i in range(repetition):
            self.forwardprop(dataset)
            self.backwardprop(dataset, answers)


# In[5]:



NN = NeuralNetwork()
print(NN.W1)
print(NN.W2)
NN.train(data, data_ans, 10000) # trains on data, provides answers to data, and amount of epochs
print(NN.W1)
print(NN.W2)
user_input_one = float(input("User Input One: "))
user_input_two = float(input("User Input Two: "))
user_input_three = float(input("User Input Three: "))
print("New Output data: ")
q = NN.forwardprop(np.array([user_input_one, user_input_two, user_input_three, 1]))
print(q)
print("Wow, we did it!")






