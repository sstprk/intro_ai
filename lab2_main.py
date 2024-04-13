#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:24:39 2024

@author: sstprk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ann import ANN
                
    
data = pd.read_excel("/Users/sstprk/Desktop/sunspot.xlsx")
data = pd.DataFrame(data)

n2x = np.array([data["Activity"][0:(data.shape[0])-2], data["Activity"][1:(data.shape[0])-1]]).T

n2y = np.array(data["Activity"][2:(data.shape[0])]).reshape((n2x.shape[0], 1))

n6x = np.array([data["Activity"][0:(data.shape[0])-6], data["Activity"][1:(data.shape[0])-5], data["Activity"][2:(data.shape[0])-4], data["Activity"][3:(data.shape[0])-3], data["Activity"][4:(data.shape[0])-2], data["Activity"][5:(data.shape[0])-1]]).T

n6y = np.array(data["Activity"][6:(data.shape[0])]).reshape((n6x.shape[0], 1))

n10x = np.array([data["Activity"][0:(data.shape[0])-10], data["Activity"][1:(data.shape[0])-9], data["Activity"][2:(data.shape[0])-8], data["Activity"][3:(data.shape[0])-7], data["Activity"][4:(data.shape[0])-6], data["Activity"][5:(data.shape[0])-5], data["Activity"][6:(data.shape[0])-4], data["Activity"][7:(data.shape[0])-3], data["Activity"][8:(data.shape[0])-2], data["Activity"][9:(data.shape[0])-1]]).T

n10y = np.array(data["Activity"][10:(data.shape[0])]).reshape(n10x.shape[0], 1)

lr2 = 1e-4
epochs2 = 10000

print("\nMODEL WITH n=2")
n2_model = ANN()
n2_model.train(n2x, n2y, lr2, epochs2)

plt.figure(0)

plt.title("n=2 MSE")

plt.plot(n2_model.mse_history[50:], color="blue")

plt.figure()

plt.title("n=2 MAD")

plt.plot(n2_model.mad_history[100:], color="green")

plt.show()

lr6 = 1e-5
epochs6 = 6000

print("\nMODEL WITH n=6")
n6_model = ANN()
n6_model.train(n6x, n6y, lr6, epochs6)

plt.figure(0)

plt.title("n=6 MSE")

plt.plot(n6_model.mse_history[100:], color="blue")

plt.figure()

plt.title("n=6 MAD")

plt.plot(n6_model.mad_history[100:], color="green")

plt.show()

lr10 = 1e-5
epochs10 = 10000

print("\nMODEL WITH n=10")
n10_model = ANN()
n10_model.train(n10x, n10y, lr10, epochs10)

plt.figure(0)

plt.title("n=10 MSE")

plt.plot(n10_model.mse_history[100:], color="blue")

plt.figure()

plt.title("n=10 MAD")

plt.plot(n10_model.mad_history[100:], color="green")

plt.show()
