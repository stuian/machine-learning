import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("Data.csv")
# 1、整个数据集以数组的形式展示
# print(dataset.values)

# 2、 切片: df.iloc[0]、df.iloc[1]、df.iloc[-1] 分别表示第一行、第二行、最后一行
#  同理df.iloc[:,0]、df.iloc[:,1]、df.iloc[:,-1] 分别表示第一列、第二列、最后一
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
# print(X)
# print(Y)

# 3、处理缺失数据
imputer = Imputer(missing_values="NaN",strategy="mean",axis = 0)
X[:,1:3] = imputer.fit_transform(X[:,1:3])
# print(X)

# 4、encode the categorical data
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
# print(X)

# 5、create a dummy variable
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).to_array() #否则会返回sparse matrix
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# 6、划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# 7、feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)