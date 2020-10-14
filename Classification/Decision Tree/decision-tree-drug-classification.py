# Importing libraries

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("drug200.csv")
# print(df.head())

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
# print(X[0:5])

# As we can see there are some categorical variable in the data set such as Sex and BP
# Unfortunately Sklearn, Decision Tree does not support Categorical data
# So we will covert categorical data to neumerical values

from sklearn import preprocessing

label_sex = preprocessing.LabelEncoder()
label_sex.fit(['F', 'M'])
X[:,1] = label_sex.transform(X[:,1])

label_BP = preprocessing.LabelEncoder()
label_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = label_BP.transform(X[:,2])
# print(X[:5])

label_Cholesterol = preprocessing.LabelEncoder()
label_Cholesterol.fit(['NORMAL', 'HIGH'])
X[:,3] = label_Cholesterol.transform(X[:,3])
# print(X[:10])

Y = df['Drug']
# print(Y[:5])

# Now we will perform train test split on the data
# for this we will use Sklearn, train_test_split from model_selection

from sklearn.model_selection import train_test_split

# Train_test_split return 4 parameters

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 3)

# now we have our train and test set
# We will build our Decision Tree
# we will pass criterian as entropy so that we will be able to see the information gain

deci_tree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
deci_tree.fit(X_train, Y_train)
predict = deci_tree.predict(X_test)
print(predict[:10])
print(Y_test[:10])

# Now as we can observe that our model is generating almost correct prediction.
# now we will check the accuracy of our models

from sklearn import metrics
import matplotlib.pyplot as plt

print('Accuracy:' , metrics.accuracy_score(Y_test, predict))

# But how this accuracy_score method works, let's see
# predicted values = ['drugY', 'drugX', 'drugY', 'drugX', 'drugX', 'drugA', 'drugY', 'drugA', 'drugB', 'drugA']
# real values = ['drugY', 'drugX', 'drugX', 'drugX', 'drugX', 'drugC', 'drugY', 'drugA', 'drugB', 'drugA']
# accuracy measure is the ratio between number of matched values and number of total values
# for the above 10 rows number of matched values are 8 hence 8/10 = accuracy is 0.8
# like wise we can calculate the accuracy using numpy's mean function as:
acc = np.mean(Y_test == predict)
print("Accuracy2: ",acc)

# now lets visualize the tree we have created 
# pip install pydotplus

from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree


# from running below script you will need to install graphviz
# windows chocolately users can install it using <choco install graphviz>
# ubuntu users can isnatall using <sudo apt-get install graphviz>
# MacOS users can install using <brew install graphviz>
# or you can download the installer from https://graphviz.org/download/ and then install it but now you
# will need to set the path as below commented code

# import os     
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

dot_data = StringIO()
filename = "decisionTree.png"
featuresName = df.columns[0:5]
target_names = df['Drug'].unique().tolist()
out = tree.export_graphviz(deci_tree, feature_names=featuresName, out_file=dot_data, class_names=np.unique(Y_train), filled=True, special_characters=True, rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100,200))
plt.imshow(img, interpolation='nearest')
plt.show()