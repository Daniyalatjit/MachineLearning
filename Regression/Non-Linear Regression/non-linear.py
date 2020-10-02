import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5.0, 5.0, 0.1)

# # We can adjust the slope and intercept to verify the changes in the graph
# y = 2*x + 3
# y_noise = 2*np.random.normal(size=x.size)
# ydata = y + y_noise

# # plt.figure(figsize=(8,6))
# plt.plot(x, ydata, 'bo')
# plt.plot(x,y, '-r')
# plt.xlabel('Dependent Variable')
# plt.ylabel('Independent Variable')
# plt.show()

# # We can adjust the slope and intercept to verify the changes in the graph
# y = x**3 + x**2 + x + 3
# y_noise = 20*np.random.normal(size=x.size)
# ydata = y + y_noise

# # plt.figure(figsize=(8,6))
# plt.plot(x, ydata, 'bo')
# plt.plot(x,y, '-r')
# plt.xlabel('Dependent Variable')
# plt.ylabel('Independent Variable')
# plt.show()

# # We can adjust the slope and intercept to verify the changes in the graph
# y = x**3 + x**2 + x + 3
# y_noise = 20*np.random.normal(size=x.size)
# ydata = y + y_noise

# # plt.figure(figsize=(8,6))
# plt.plot(x, ydata, 'bo')
# plt.plot(x,y, '-r')
# plt.xlabel('Dependent Variable')
# plt.ylabel('Independent Variable')
# plt.show()

# # Quadretic
# y = np.power(x,2)
# y_noise = 20*np.random.normal(size=x.size)
# ydata = y + y_noise

# # plt.figure(figsize=(8,6))
# plt.plot(x, ydata, 'bo')
# plt.plot(x,y, '-r')
# plt.xlabel('Dependent Variable')
# plt.ylabel('Independent Variable')
# plt.show()

# # Exponential
# y = np.exp(x)
# y_noise = 20*np.random.normal(size=x.size)
# ydata = y + y_noise

# # plt.figure(figsize=(8,6))
# plt.plot(x, ydata, 'bo')
# plt.plot(x,y, '-r')
# plt.xlabel('Dependent Variable')
# plt.ylabel('Independent Variable')
# plt.show()

# # Logrithmic
# y = np.log(x)
# y_noise = 20*np.random.normal(size=x.size)
# ydata = y + y_noise

# # plt.figure(figsize=(8,6))
# plt.plot(x, ydata, 'bo')
# plt.plot(x,y, '-r')
# plt.xlabel('Dependent Variable')
# plt.ylabel('Independent Variable')
# plt.show()

import pandas as pd

df = pd.read_csv("china_gdp.csv")
# print(df.describe())
print(df.head())