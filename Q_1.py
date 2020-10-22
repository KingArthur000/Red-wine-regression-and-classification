import pandas as pd
#import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

train = pd.read_csv("wineQualityRed_train.csv")
test = pd.read_csv("wineQualityRed_test.csv")

print(train.head())

y_train = train[['quality']]
X_train = train[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
              'total sulfur dioxide','density','pH','sulphates','alcohol']]
y_test = test[['quality']]
X_test = test[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
              'total sulfur dioxide','density','pH','sulphates','alcohol']]

x_train = normalize(X_train)
x_test = normalize(X_test)

weights = LinearRegression().fit(x_train,y_train)
y_pred = weights.predict(X_test)
print("Score: ",weights.score(x_train,y_train))

print("Weights: \n",weights.coef_)
print("\nIntercept: \n",weights.intercept_)
print("\nMean squared error: \n", mean_squared_error(y_test,y_pred))

x_plot = train[['sulphates']]
weight = LinearRegression().fit(x_plot,y_train)
plt.plot(x_plot,weight.coef_*x_plot+weight.intercept_)
plt.scatter(x_plot,y_train,c = 'r',marker = 'o')
plt.show()
