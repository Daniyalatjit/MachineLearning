import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing 

df = pd.read_csv("telecust1000t.csv")
# print(df.head())
# print(df.describe())
# print(df['custcat'].value_counts())

# Exploring data using visualaizati0on technique
# plt.hist(df['income'], bins=50)
# plt.show()

# print(df.columns)

x = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values
# print(x[:5])

y = df['custcat'].values
#print(y[0:5])

# Standarizing Data
x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))
# print(x[0:5])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=4)
print('Train Set: ', X_train.shape, Y_train.shape)
print("Test Set: ", X_test.shape, Y_test.shape)

# K-Nearest Neighbors (K-NN)
# importing the libraries

from sklearn.neighbors import KNeighborsClassifier

# k = 6 # try on 2, 3, 4, 5, 6 and compare accuracies and the select the optimum value of k

# # On hit and trial we found that:
# # -----------------------------------------------------------------------------
# #      Case       |  k = 1   |  k = 2  | k = 3   | k = 4  | k = 5  |  k = 6   |
# # -----------------------------------------------------------------------------
# # Train Accuracy: |   1.0    |  0.6175 | 0.56875 | 0.5475 | 0.5375 |  0.51625 |
# # Test Accuracy : |   0.3    |  0.29   | 0.315   | 0.32   | 0.32   |  0.31    |
# # -----------------------------------------------------------------------------
# # In above observation we can see that best accuracy is at k = 4

k = 4
# train model and predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
print(neigh)

yhat = neigh.predict(X_test)
print("\nPredicted(5-rows): ",yhat[:5])
print("Actual(5-rows): ", Y_test[:5])

# Accuracy Evaluation
from sklearn import metrics
print('Train set accuracy: ', metrics.accuracy_score(Y_train, neigh.predict(X_train)))
# out-of-sample test
print('Test set Accuracy: ', metrics.accuracy_score(Y_test, yhat))
# Accuracy Evaluation
from sklearn import metrics
print('Train set accuracy: ', metrics.accuracy_score(Y_train, neigh.predict(X_train)))
# out-of-sample test
print('Test set Accuracy: ', metrics.accuracy_score(Y_test, yhat))