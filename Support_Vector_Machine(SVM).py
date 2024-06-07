import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('D:\\Datasets\\Social_Network_Ads.csv')

x = data.iloc[:, [2, 3]]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

classifier = SVC(kernel='linear', random_state=0)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')

plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)

from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
plt.scatter(x_test[:,0], x_test[:,1], c=y_test)

#Creation on hyperplane
w = classifier.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-2.5,2.5)
yy = a*xx-(classifier.intercept_[0])/w[1]
plt.plot(xx,yy)
plt.show()
