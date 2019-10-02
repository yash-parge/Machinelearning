#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
#imports dataset
data = pd.read_csv(
    r"Input file",
    usecols=[],
    encoding = "utf-8",
    dtype={}
)
data['Bias'] = 0.0 # Adds bias coloumn to dataset
def standard(df, num):
    df = df.copy(deep=True)
    for i in range(num):
        series = df.iloc[:,i]
        avg = series.mean()
        stdv = series.std()
        standardised = (series - avg)/stdv
        df.iloc[:,i] = standardised
    return df

def sig(df, num):
    df = df.copy(deep=True)
    for i in range(len(df)):
        df.iloc[i, num] = 1/(1+np.exp(-(df.iloc[i, num])))
    return df

print(data)
data = standard(data, 4)
data_valid = data
data_valid = data_valid.tail(1000)
data = data[:-1000]
print(data)
data = data.to_numpy()
#imports the results coloumn of the dataset
data_ans = pd.read_csv(
    r"Answer file",
    usecols=[],
    encoding = "utf-8",
    dtype={}
)
data_ans = standard(data_ans, 1)
data_ans = sig(data_ans, 0)
data_ans_valid = data_ans
data_ans_valid = data_ans_valid.tail(1000)
data_ans = data_ans[:-1000]
print(data_ans)
data_ans = data_ans.to_numpy()
data_valid = data_valid.reset_index(drop=True)
data_ans_valid = data_ans_valid.reset_index(drop=True)
# In[2]:


class NeuralNetwork(object):
    def __init__(self):
        self.inputsize = 5 + 1 #Number of inputs given + bias node
        self.hiddensize = 10 #Number of nodes in the hidden layer
        self.outputsize = 1 #Number of nodes in the output layer
        np.random.seed(1)
        #self.bias_2 = np.random.randn(self.hiddensize)
        self.W1 = (np.random.randn(self.inputsize, self.hiddensize))/np.sqrt(self.hiddensize)
        self.W2 = (np.random.randn(self.hiddensize, self.outputsize))/np.sqrt(self.hiddensize)
        
    def forwardprop(self, input):
        self.input = input
        self.z1 = np.dot(input, self.W1) #Dot function of inputs and out first weights
        #self.z1 = self.z1 + self.bias_2 # Adds bias
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
        egg = ((0.01 * self.outputerror) * self.sigmoid(self.out, True))
        weight1 = np.dot(self.input.T, (np.dot(egg, self.W2.T)) * self.sigmoid(self.z1activ, True)) # Calculates error in weight 1
        weight2 = np.dot(self.z1activ.T, egg) # Calculates error in weight 2
        self.W1 += weight1 # updates weight 1
        self.W2 += weight2 # updates weight 2
        #self.bias_2 += (0.01 * np.sum(weight1, axis=0)) # updates bias
        
    def train (self, dataset, answers, repetition):
        for i in range(repetition):
            self.forwardprop(dataset)
            self.backwardprop(dataset, answers)


# In[5]:



NN = NeuralNetwork()
print(NN.W1)
print(NN.W2)
NN.train(data, data_ans, 100000) # trains on data, provides answers to data, and amount of epochs
print(NN.W1)
print(NN.W2)

df3 = pd.DataFrame()

open(r"File to write to", 'w').close()

for i in range(len(data_valid)):
    r = NN.forwardprop(np.array([], 0.0]))
    dfAppend = pd.DataFrame({})
    df3 = df3.append(dfAppend)
df3.to_csv(r"file to write to", mode='w', header=True)



print("Wow, we did it!")

