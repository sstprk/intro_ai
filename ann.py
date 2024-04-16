#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 23:27:42 2024

@author: sstprk
"""
import numpy as np
from sklearn.model_selection import train_test_split

class ANN:
    def __init__(self):
        self.weights = None
        self.biases = None
        self.mse_history_train = []
        self.mad_history_train = []
        
        self.mse_history_valid = []
        self.mad_history_valid = []
        
        self.mse_history_test = []
        self.mad_history_test = []
        
        self.initial_w = None
        self.initial_b = None
       
        self.x_train = []
        self.y_train = []
        self.x_valid = []
        self.y_valid = []
        self.x_test = []
        self.y_test = []
            
    def init_params(self, shape_x, shape_y):
        np.random.seed(0)
        self.weights = np.random.random((1, shape_y)).T
        self.biases = np.random.random((1, 1))
        self.initial_w, self.initial_b = self.weights.copy(), self.biases.copy()
        
            
    def split_data(self, data_x, data_y, train_rate=0.7):
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=train_rate, random_state=1)
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
            
    def train(self, train_x, train_y, lr, epochs):
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = self.split_data(train_x, train_y)
        
        self.init_params(self.x_train.shape[0], self.x_train.shape[1])
        
        for i in range(epochs):
            predictions = self.forward(self.x_train)
            grad = self.mse_grad(predictions, self.y_train)
            
            self.mse_history_train.append(self.mse(predictions, self.y_train))
            self.mad_history_train.append(self.mad(predictions, self.y_train))
            
            self.backward(self.x_train, lr, grad)
            
            if i % (epochs // 10) == 0:
                predictions_valid = self.forward(self.x_valid)
                grad_valid = self.mse_grad(predictions_valid, self.y_valid)
                
                self.backward(self.x_valid, lr, grad_valid)
                
                predictions_test = self.forward(self.x_test)

                mse_valid = self.mse(predictions_valid, self.y_valid)
                
                mse_test = self.mse(predictions_test, self.y_test)
                
                mad_valid = self.mad(predictions_valid, self.y_valid)
                
                mad_test = self.mad(predictions_test, self.y_test)
                
                self.mse_history_valid.append(mse_valid)
                self.mad_history_valid.append(mad_valid)
                
                self.mse_history_test.append(mse_test)
                self.mad_history_test.append(mad_test)


                print("-------------------------------------------------------")
                print(f"Epoch {i} MSE Validation: {mse_valid} -- MAD: {mad_valid}")
                print(f"Epoch {i} MSE Test: {mse_test} -- MAD:{mad_test} ")