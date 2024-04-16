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

plt.figure()

plt.title("Sun Plum Activity")

plt.xlabel("Year")
plt.ylabel("Activity")

plt.plot(data["Year"], data["Activity"])

plt.show()

n2x = np.array([data["Activity"][0:(data.shape[0])-2], data["Activity"][1:(data.shape[0])-1]]).T

n2y = np.array(data["Activity"][2:(data.shape[0])]).reshape((n2x.shape[0], 1))

n6x = np.array([data["Activity"][0:(data.shape[0])-6], data["Activity"][1:(data.shape[0])-5], data["Activity"][2:(data.shape[0])-4], data["Activity"][3:(data.shape[0])-3], data["Activity"][4:(data.shape[0])-2], data["Activity"][5:(data.shape[0])-1]]).T

n6y = np.array(data["Activity"][6:(data.shape[0])]).reshape((n6x.shape[0], 1))

n10x = np.array([data["Activity"][0:(data.shape[0])-10], data["Activity"][1:(data.shape[0])-9], data["Activity"][2:(data.shape[0])-8], data["Activity"][3:(data.shape[0])-7], data["Activity"][4:(data.shape[0])-6], data["Activity"][5:(data.shape[0])-5], data["Activity"][6:(data.shape[0])-4], data["Activity"][7:(data.shape[0])-3], data["Activity"][8:(data.shape[0])-2], data["Activity"][9:(data.shape[0])-1]]).T

n10y = np.array(data["Activity"][10:(data.shape[0])]).reshape(n10x.shape[0], 1)


#n=2 Model
lr2 = 1e-4
epochs2 = 10000

print("\nMODEL WITH n=2")
n2_model = ANN()
n2_model.train(n2x, n2y, lr2, epochs2)

#Training results
plt.figure(0, figsize=(10, 8))

plt.subplot(2,1,1)
plt.title("n=2 MSE Train")

plt.plot(n2_model.mse_history_train[100:], color="blue")

plt.subplot(2,1,2)
plt.title("n=2 MAD Train")

plt.plot(n2_model.mad_history_train[100:], color="green")

#Validation results
plt.figure(1, figsize=(10, 8))

plt.subplot(2,1,1)
plt.title("n=2 MSE Validation")

plt.plot(n2_model.mse_history_valid, color="blue")

plt.subplot(2,1,2)
plt.title("n=2 MAD Validation")

plt.plot(n2_model.mad_history_valid, color="green")

#Test results
plt.figure(3, figsize=(10, 8))

plt.subplot(2,1,1)
plt.title("n=2 MSE Test")

plt.plot(n2_model.mse_history_test, color="blue")

plt.subplot(2,1,2)
plt.title("n=2 MAD Test")

plt.plot(n2_model.mad_history_test, color="green")

plt.show()


#n=6 Model
lr6 = 1e-5
epochs6 = 6000

print("\nMODEL WITH n=6")
n6_model = ANN()
n6_model.train(n6x, n6y, lr6, epochs6)

#Training results
plt.figure(4, figsize=(10, 8))

plt.subplot(2,1,1)
plt.title("n=6 MSE Train")

plt.plot(n6_model.mse_history_train[100:], color="blue")

plt.subplot(2,1,2)
plt.title("n=6 MAD Train")

plt.plot(n6_model.mad_history_train[100:], color="green")

#Validation results
plt.figure(5, figsize=(10, 8))

plt.subplot(2,1,1)
plt.title("n=6 MSE Validation")

plt.plot(n6_model.mse_history_valid, color="blue")

plt.subplot(2,1,2)
plt.title("n=6 MAD Validation")

plt.plot(n6_model.mad_history_valid, color="green")

#Test results
plt.figure(6, figsize=(10, 8))

plt.subplot(2,1,1)
plt.title("n=6 MSE Test")

plt.plot(n6_model.mse_history_test, color="blue")

plt.subplot(2,1,2)
plt.title("n=6 MAD Test")

plt.plot(n6_model.mad_history_test, color="green")

plt.show()


#n=10 Model
lr10 =2e-5
epochs10 = 10000

print("\nMODEL WITH n=10")
n10_model = ANN()
n10_model.train(n10x, n10y, lr10, epochs10)

#Training results
plt.figure(7, figsize=(10, 8))

plt.subplot(2,1,1)
plt.title("n=10 MSE Train")

plt.plot(n10_model.mse_history_train[100:], color="blue")

plt.subplot(2,1,2)
plt.title("n=10 MAD Train")

plt.plot(n10_model.mad_history_train[100:], color="green")

#Validation results
plt.figure(8, figsize=(10, 8))

plt.subplot(2,1,1)
plt.title("n=10 MSE Validation")

plt.plot(n10_model.mse_history_valid, color="blue")

plt.subplot(2,1,2)
plt.title("n=10 MAD Validation")

plt.plot(n10_model.mad_history_valid, color="green")

#Test results
plt.figure(9, figsize=(10, 8))

plt.subplot(2,1,1)
plt.title("n=10 MSE Test")

plt.plot(n10_model.mse_history_test, color="blue")

plt.subplot(2,1,2)
plt.title("n=10 MAD Test")

plt.plot(n10_model.mad_history_test, color="green")

plt.show()
