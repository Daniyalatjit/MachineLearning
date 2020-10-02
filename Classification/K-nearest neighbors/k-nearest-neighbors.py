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
