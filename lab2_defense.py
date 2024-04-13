#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:05:26 2024

@author: sstprk
"""
import numpy as np

def init_params():
    x_train = np.array([[1, 2],
                        [4, 5],
                        [7, 8]])
    
    y_train = np.array([3, 6, 9]).T
    
    y_train = np.reshape(y_train, (3,1))
    
    weights = np.array([0.1, 0.2]).reshape((2,1))
    
    bias = 0.3
    
    return weights, bias, x_train, y_train

def forward_prop(weights, bias, x_train, y_train):
    #Making predictions
    return np.dot(x_train, weights) + bias

def act_func(x):
    return 2*x

def backward_prop(train, actual, predictions, lr, weights, bias):
    #Gradient descent is the derivative of act function.
    grad = 2
    
    #Updating the weights and bias
    mean = np.mean(train)
    w_update = np.mean(train) * 2
    b_update = 2
    
    weights -= w_update * lr
    bias -= b_update * lr
    
    return weights, bias
    
if __name__ == "__main__":
    weights, bias, x_train, y_train = init_params()
    
    lr = 0.01
    predictions = forward_prop(weights, bias, x_train, y_train).reshape((3,1))
    
    new_weights, new_bias = backward_prop(x_train, y_train, predictions, lr, weights, bias)
    
    new_predictions = forward_prop(new_weights, new_bias, x_train, y_train)