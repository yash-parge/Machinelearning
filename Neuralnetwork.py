import numpy as np
import pandas as pd

#Imports the dataset to train 
data = pd.read_csv(
    "./Dataset/Appleinfobak.csv",
    usecols=['Open', 'High', 'Low', 'Volume'],
    dtype={"Open": float, "High": float, "Low": float, "Volume": float}
)
#imports the answers to the dataset which we will set as the closing price
data_ans = pd.read_csv(
    "./Dataset/Appleinfobak.csv",
    usecols=['Close'],
    dtype={"Close": float}
)
#Creates the neural network object
class NeuralNetwork(object):
    def __init__(self):
        self.inputsize = 4 #Number of inputs given
        self.hiddensize = 6 #Number of nodes in the hidden layer
        self.outputsize = 1 #Number of nodes in the output layer
        np.random.seed(1)

        self.W1 = np.random.randn(self.inputsize, self.hiddensize) #Creates weights
        self.W2 = np.random.randn(self.hiddensize, self.outputsize)
    
    def forwardprop(self, input):
        self.input = input
        self.z1 = np.dot(input, self.W1) #Dot function of inputs and out first weights
        self.z1activ = self.sigmoid(self.z1) #Activation of hidden layer outputs
        self.z2 = np.dot(self.z1activ, self.W2) #Dot function of hidden layer and our second weights
        self.out = self.sigmoid(self.z2) #Activation of output layer
        
        return self.out
    
    #Sigmoid and sigmoid derivitive functions
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1-x)
        return 1 / (1 + np.exp(-x))
    
    def backwardprop(self, input, answers):
        self.outputerror = answers - self.out #This will be our loss function
        #Calculates the impact of weights on the final output and readjusts them accordingly using the chain rule
        egg = ((2 * self.outputerror) * self.sigmoid(self.out, True)) 
        weight2 = np.dot(self.z1activ.T, egg)
        weight1 = np.dot(self.input.T, (np.dot(egg, self.W2.T)) * self.sigmoid(self.z1activ, True))
        self.W1 += weight1
        self.W2 += weight2

    def train (self, dataset, answers, repetition): #Creates a loop to propogate forward and adjust weights
        for i in range(repetition):
            self.forwardprop(dataset)
            self.backwardprop(dataset, answers)
    
NN = NeuralNetwork() #Creates our Neural network
print(NN.W1)
print(NN.W2) # Prints inital starting weights
NN.train(data, data_ans, 100)
print(NN.W1)
print(NN.W2) #Prints weightage after training
print(NN.forwardprop(data))
