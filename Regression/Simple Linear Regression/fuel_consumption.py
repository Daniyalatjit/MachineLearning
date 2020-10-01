import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("FuelConsumptionCo2.csv")

# Showing head of the Data 
# print(df.head())

# Observing descriptive analysis of Data 
# print(df.describe())

# Looking deeper in the Data
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
# viz.hist()
# plt.show()

# Scatter Plot between FUELCONSUMPTIONO_COMB and CO2EMISSION
# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='red')
# plt.xlabel("Fuel Consumption Comb.")
# plt.ylabel("Co2 Emission")
# plt.show()

# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS)
# plt.xlabel("Engine Size")
# plt.ylabel("Co2 Emission")
# plt.show()

# Splitting the data into train and test data frames actualizing truly out-of-sample testing
mask = np.random.rand(len(df)) < 0.8
train = df[mask]
test = df[~mask]

# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
# plt.ylabel("Co2 Emission")
# plt.xlabel("Engine Size")
# plt.show()

# CREATING THE MODEL USING SKLEARN

from sklearn import linear_model
reg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
reg.fit(train_x, train_y)

# The coefficients
print("Coefficient: ", reg.coef_)
print("Intercept: ", reg.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
plt.plot(train_x, reg.coef_[0][0]*train_x + reg.intercept_[0], '-r')
plt.xlabel('Engin Size')
plt.ylabel('Co2 Emission')
plt.show()


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = reg.predict(test_x)

print("Mean Absolute Error: %.2f" %np.mean(np.absolute(test_y_ - test_y)))
print("Residule sum of squares: %.2f" %np.mean((test_y_-test_y)**2))
print("R2 Score: %.2f" %r2_score(test_y, test_y_))