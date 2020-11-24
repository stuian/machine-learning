import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# from sklearn.svm import SVC
# classifier = SVC(kernel = 'linear', random_state = 0)
# classifier.fit(X_train, y_train)

d = datasets.load_iris()
x = d.data
y = d.target
x=x[y<2,:2]
y=y[y<2]
# print(x)
# print(y)
plt.figure()
plt.scatter(x[y==0,0],x[y==0,1],color="r")
plt.scatter(x[y==1,0],x[y==1,1],color="g")
plt.show()

sc = StandardScaler()
sc.fit(x)
x = sc.transform(x)
sm = LinearSVC()
sm.fit(x,y)