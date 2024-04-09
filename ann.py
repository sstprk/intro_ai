#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 23:27:42 2024

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
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = self.split_data(train_x, train_y)
        
        self.init_params(self.x_train.shape[0], self.x_train.shape[1])
        
        for i in range(epochs):
            predictions = self.forward(self.x_train)
            grad = self.mse_grad(predictions, self.y_train)
            
            self.backward(self.x_train, lr, grad)
            
            if i % print_oneach == 0:
                predictions = self.forward(self.x_valid)
                
                mse = self.mse(predictions, self.y_valid)
                self.mse_history.append(mse)
                
                mad = self.mad(predictions, self.y_valid)
                self.mad_history.append(mad)
                
                print(f"Epoch {i} MSE: {mse} -- MAD: {mad}")
                
    def test(self):
        predictions = self.forward(self.x_test)
        accuracy_mse = self.mse(predictions, self.y_test)
        
        return predictions, self.y_test, accuracy_mse
        