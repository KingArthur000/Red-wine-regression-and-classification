import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train = pd.read_csv("wineQualityRed_train.csv")
test = pd.read_csv("wineQualityRed_test.csv")

y_train = np.array(train[['quality']]>=7)*1
y_train = y_train.reshape(len(y_train),)
X_train = train[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
              'total sulfur dioxide','density','pH','sulphates','alcohol']]
y_test = np.array(test[['quality']]>=7)*1
X_test = test[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
              'total sulfur dioxide','density','pH','sulphates','alcohol']]

x_train = normalize(X_train)
x_test = normalize(X_test)

weights = LinearRegression(fit_intercept=False).fit(x_train,y_train)
y_pred = weights.predict(x_test)
print('Score: ',accuracy_score((y_pred==1),(y_test==1)))

test[['class']] = np.where(test[['quality']]>=7, 'Good', 'Bad')
test[['predict']] = np.where((y_pred==1), 'Good', 'Bad')

fig, (ax1, ax2) = plt.subplots(2)

test['class'].value_counts().plot(kind='bar',figsize=(10,8),title="Actual Quality of Red Wine",ax = ax1)
print("y_test: \n",test['class'].value_counts())

test['predict'].value_counts().plot(kind='bar',figsize=(10,8),title="Predicted Quality of Red Wine",ax = ax2)
print("y_pred: \n",test['predict'].value_counts())

plt.show()
