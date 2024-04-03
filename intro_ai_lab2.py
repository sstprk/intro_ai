#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:24:39 2024

@author: sstprk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class ANN:
    def __init__(self):
        self.weights = None
        self.biases = None
        self.mse_history = []
        self.mad_history = []
            
    def init_params(self, shape_x, shape_y):
        np.random.seed(0)
        self.weights = np.random.random((1, shape_y)).T
        self.biases = np.random.random((1, 1))
            
    def split_data(self, data_x, data_y):
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.7, random_state=1)
        x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, train_size=0.5, random_state=1)
        return x_train, y_train, x_valid, y_valid, x_test, y_test
        
    def forward(self, train_data_x):
        return np.dot(train_data_x, self.weights) + self.biases
    
    def mse(self, predictions, actual):
        return np.mean((predictions - actual)**2)
    
    def mad(self, predictions, actual):
        return np.median(np.abs(predictions - actual))
    
    def mse_grad(self, predictions, actual):
        return predictions - actual
     
    def backward(self, train_x, lrate, grad):
        n = train_x.shape[0]
        w_grad = np.dot((train_x.T / n), grad)     
        b_grad = np.mean(grad, axis=0)
        
        self.weights -= w_grad * lrate
        self.biases -= b_grad * lrate
            
    def train(self, train_x, train_y, lr, epochs, print_oneach=500):
        x_train, y_train, x_valid, y_valid, x_test, y_test = self.split_data(train_x, train_y)
        
        self.init_params(x_train.shape[0], x_train.shape[1])
        
        for i in range(epochs):
            predictions = self.forward(x_train)
            grad = self.mse_grad(predictions, y_train)
            
            self.backward(x_train, lr, grad)
            
            if i % print_oneach == 0:
                predictions = self.forward(x_valid)
                
                mse = self.mse(predictions, y_valid)
                self.mse_history.append(mse)
                
                mad = self.mad(predictions, y_valid)
                self.mad_history.append(mad)
                
                print(f"Epoch {i} MSE: {mse} -- MAD: {mad}")
                
    
data = pd.read_excel("/Users/sstprk/Desktop/sunspot.xlsx")
data = pd.DataFrame(data)

n2x = np.array([data["Activity"][0:(data.shape[0])-2], data["Activity"][1:(data.shape[0])-1]]).T

n2y = np.array(data["Activity"][2:(data.shape[0])]).reshape((n2x.shape[0], 1))

n6x = np.array([data["Activity"][0:(data.shape[0])-6], data["Activity"][1:(data.shape[0])-5], data["Activity"][2:(data.shape[0])-4], data["Activity"][3:(data.shape[0])-3], data["Activity"][4:(data.shape[0])-2], data["Activity"][5:(data.shape[0])-1]]).T

n6y = np.array(data["Activity"][6:(data.shape[0])]).reshape((n6x.shape[0], 1))

n10x = np.array([data["Activity"][0:(data.shape[0])-10], data["Activity"][1:(data.shape[0])-9], data["Activity"][2:(data.shape[0])-8], data["Activity"][3:(data.shape[0])-7], data["Activity"][4:(data.shape[0])-6], data["Activity"][5:(data.shape[0])-5], data["Activity"][6:(data.shape[0])-4], data["Activity"][7:(data.shape[0])-3], data["Activity"][8:(data.shape[0])-2], data["Activity"][9:(data.shape[0])-1]]).T

n10y = np.array(data["Activity"][10:(data.shape[0])]).reshape(n10x.shape[0], 1)

lr2 = 1e-4
epochs2 = 6000
print_oneach2 = 500

n2_model = ANN()
n2_model.train(n2x, n2y, lr2, epochs2, print_oneach2)
print("Model with n=2")

plt.figure(0)
plt.legend(("mse", "mad"))

plt.title("n=2")

plt.plot(n2_model.mse_history, color="blue")
plt.plot(n2_model.mad_history, color="green")

plt.show()

lr6 = 1e-5
epochs6 = 6000
print_oneach6 = 500


n6_model = ANN()
n6_model.train(n6x, n6y, lr6, epochs6, print_oneach6)
print("Model with n=6")

plt.figure(0)
plt.legend(("mse", "mad"))

plt.title("n=6")

plt.plot(n6_model.mse_history, color="blue")
plt.plot(n6_model.mad_history, color="green")

plt.show()

lr10 = 6e-5
epochs10 = 20000
print_oneach10 = 2500

n10_model = ANN()
n10_model.train(n10x, n10y, lr10, epochs10, print_oneach10)
print("Model with n=10")

plt.figure(0)
plt.legend(("mse", "mad"))

plt.title("n=10")

plt.plot(n10_model.mse_history, color="blue")
plt.plot(n10_model.mad_history, color="green")

plt.show()
