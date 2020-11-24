import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1、data preprocessing
dataset = pd.read_csv("studentscores.csv")
# print(dataset)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=1)

# 2、fitting Simple Linear Regression Model to the training set
linear_regression = LinearRegression()
regressor = linear_regression.fit(X_train,Y_train)

# 3、Predecting the Result
y_pred = regressor.predict(X_test)

# 4、Visualization
# Visualising the Training results
plt.scatter(X_train,Y_train,color = "red")
# plt.show()

plt.plot(X_train,regressor.predict(X_train),color = "blue")
plt.show()

plt.scatter(X_test,Y_test,color = "red")
plt.plot(X_test , y_pred, color ='blue')
plt.show()