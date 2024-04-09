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
epochs2 = 6000
print_oneach2 = 500

n2_model = ANN()
n2_model.train(n2x, n2y, lr2, epochs2, print_oneach2)

predictions2, actual2, mse2 = n2_model.test()
print("Model with n=2")

plt.figure(0)
plt.legend(("mse", "mad"))

plt.title("n=2")

plt.plot(n2_model.mse_history, color="blue")
plt.plot(n2_model.mad_history, color="green")

plt.figure()

plt.title(f"{mse2}")
plt.legend("actual", "predicted")

plt.plot(predictions2, color="red")
plt.plot(actual2, color="blue")

plt.show()

lr6 = 1e-5
epochs6 = 6000
print_oneach6 = 500


n6_model = ANN()
n6_model.train(n6x, n6y, lr6, epochs6, print_oneach6)

predictions6, actual6, mse6 = n6_model.test()
print("Model with n=6")

plt.figure(0)
plt.legend(("mse", "mad"))

plt.title("n=6")

plt.plot(n6_model.mse_history, color="blue")
plt.plot(n6_model.mad_history, color="green")

plt.figure()

plt.title(f"{mse6}")
plt.legend("actual", "predicted")

plt.plot(predictions6, color="red")
plt.plot(actual6, color="blue")

plt.show()

lr10 = 1e-5
epochs10 = 20000
print_oneach10 = 2500

n10_model = ANN()
n10_model.train(n10x, n10y, lr10, epochs10, print_oneach10)

predictions10, actual10, mse10 = n10_model.test()
print("Model with n=10")

plt.figure(0)
plt.legend(("mse", "mad"))

plt.title("n=10")

plt.plot(n10_model.mse_history, color="blue")
plt.plot(n10_model.mad_history, color="green")

plt.figure()

plt.title(f"{mse10}")
plt.legend("actual", "predicted")

plt.plot(predictions10, color="red")
plt.plot(actual10, color="blue")

plt.show()
