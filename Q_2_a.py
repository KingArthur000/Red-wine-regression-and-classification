import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("wineQualityRed_train.csv")
test = pd.read_csv("wineQualityRed_test.csv")

y_train = np.array(train[['quality']]>=7)
y_train = y_train.reshape(len(y_train),)
x_train = train[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
              'total sulfur dioxide','density','pH','sulphates','alcohol']]
y_test = np.array(test[['quality']]>=7)
y_test = y_test.reshape(len(y_test),)
x_test = test[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
              'total sulfur dioxide','density','pH','sulphates','alcohol']]

logistic = LogisticRegression(solver='saga',max_iter=10000).fit(x_train, y_train)
print("Score: ",logistic.score(x_test,y_test))

test[['class']] = np.where(test[['quality']]>=7, 'Good', 'Bad')
test[['predict']] = np.where(logistic.predict(x_test), 'Good', 'Bad')

fig, (ax1, ax2) = plt.subplots(2)

test['class'].value_counts().plot(kind='bar',figsize=(10,8),title="Actual Quality of Red Wine",ax = ax1)
print("y_test: \n",test['class'].value_counts())

test['predict'].value_counts().plot(kind='bar',figsize=(10,8),title="Predicted Quality of Red Wine",ax = ax2)
print("y_pred: \n",test['predict'].value_counts())

plt.show()
