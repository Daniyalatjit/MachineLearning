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
# print(df.head())

# plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
# plt.plot(x_data, y_data, 'ro')
# plt.xlabel("Year")
# plt.ylabel("GDP")
# plt.show()

# here we can see the curve intially grows with a low slope, then a rapid growth and at the end we can see a slight decrease in the GDP.
# so accordingly we can use logistic function, it fits the best

# we can observe it here
# x = np.arange(-5.0, 5.0, 0.1)
# y = 1 / (1 + np.exp(-x))
# plt.plot(x,y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y

beta_1 = 0.10
beta_2 = 1990.0

# logistic function
y_pred = sigmoid(x_data, beta_1, beta_2)

# initial prediction 
# plt.plot(x_data, y_pred*15000000000000.)
# plt.plot(x_data, y_data, 'ro')
# plt.show()

# normalizing data
xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

# calculating best parameters for our models
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
# final Parameters are
print("beta_1: %f\nbeta_2: %f"%(popt[0], popt[1]))

# ploting final non-linear regression plot
a = np.linspace(1960, 2015, 55)
a = a/max(a)
plt.figure(figsize=(8,5))
b = sigmoid(a, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(a,b, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# Calculating error
# split data into train/test

from sklearn.metrics import r2_score
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )

plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(a,b, linewidth=3.0, label='predicted', color="blue")
plt.plot(test_x, test_y, linewidth=3.0, label='Actual', color="green")
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


