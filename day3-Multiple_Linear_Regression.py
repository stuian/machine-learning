import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1、Data Preprocessing
dataset = pd.read_csv("50_Startups.csv")
print(dataset)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# Encoding Categorical data
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray() # 虚拟变量被放到最前面
# Avoiding Dummy Variable Trap
X = X[:,1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# 2、Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor = regressor.fit(X_train,Y_train)

y_pred = regressor.predict(X_test)