#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd

data = pd.read_csv(
    "C:/Users/yash_/OneDrive/Documents/Development/Dataset/trial.csv",
    usecols=['1', '2', '3'],
    dtype={"1": int, "2": int, "3": int}
)
#imports the answers to the dataset which we will set as the closing price
data_ans = pd.read_csv(
    "C:/Users/yash_/OneDrive/Documents/Development/Dataset/trial.csv",
    usecols=['Answers'],
    dtype={"Answers": int}
)


# In[2]:


#Creates the neural network object
class NeuralNetwork(object):
    def __init__(self):
        self.inputsize = 3 #Number of inputs given
        self.hiddensize = 6 #Number of nodes in the hidden layer
        self.outputsize = 1 #Number of nodes in the output layer
        np.random.seed(1)

        self.W1 = np.random.randn(self.inputsize, self.hiddensize)
        self.W2 = np.random.randn(self.hiddensize, self.outputsize)
    
    def forwardprop(self, input):
        self.input = input
        self.z1 = np.dot(input, self.W1) #Dot function of inputs and out first weights
        self.z1activ = self.sigmoid(self.z1) #Activation of hidden layer outputs
        self.z2 = np.dot(self.z1activ, self.W2) #Dot function of hidden layer and our second weights
        self.out = self.sigmoid(self.z2)
        
        return self.out
    
    #Sigmoid and sigmoid derivitive functions
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1-x)
        return 1 / (1 + np.exp(-x))
    
    def backwardprop(self, input, answers):
        self.outputerror = answers - self.out #This will be our loss function
        egg = ((2 * self.outputerror) * self.sigmoid(self.out, True))
        weight2 = np.dot(self.z1activ.T, egg)
        weight1 = np.dot(self.input.T, (np.dot(egg, self.W2.T)) * self.sigmoid(self.z1activ, True))
        self.W1 += weight1
        self.W2 += weight2

    def train (self, dataset, answers, repetition):
        for i in range(repetition):
            self.forwardprop(dataset)
            self.backwardprop(dataset, answers)


# In[5]:


NN = NeuralNetwork()
print(NN.W1)
print(NN.W2)
NN.train(data, data_ans, 1000)
print(NN.W1)
print(NN.W2)
print(NN.forwardprop(data))
user_input_one = int(input("User Input One: "))
user_input_two = int(input("User Input Two: "))
user_input_three = int(input("User Input Three: "))
#user_input_four = str(input("User Input Three: "))
print("Considering New Situation: ", user_input_one, user_input_two, user_input_three)
print("New Output data: ")
print(NN.forwardprop(np.array([user_input_one, user_input_two, user_input_three])))
print("Wow, we did it!")


# In[ ]:




