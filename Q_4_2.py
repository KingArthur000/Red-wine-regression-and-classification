import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv("wineQualityRed_train.csv")
test = pd.read_csv("wineQualityRed_test.csv")

y_train = np.array(train[['quality']]>=7)
y_train = y_train.reshape(len(y_train),)
X_train = train[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
              'total sulfur dioxide','density','pH','sulphates','alcohol']]
y_test = np.array(test[['quality']]>=7)
y_test = y_test.reshape(len(y_test),)
X_test = test[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
              'total sulfur dioxide','density','pH','sulphates','alcohol']]
y = test[['quality']]>=7

pca = PCA(n_components=4)
x_train = pca.fit_transform(X_train)
x_test = pca.transform(X_test)
print("New dimensions of X_train: ",np.shape(x_train))
print("New dimensions of X_test: ",np.shape(x_test))

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

logistic = LogisticRegression(solver='saga',max_iter=10000,class_weight = {True:.8,False:.2}).fit(x_train, y_train)
y_l = logistic.predict(x_test)

print('\n**Measures for Logistic Regression**')

tp,fp,fn,tn = confusion_matrix(y,y_l).ravel()
sensitivity = tp/(tp+fp)
specificity = tn/(fn+tn)

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print('Accuracy:    ',accuracy_score(y,y_l))
print('Precision:   ',precision)
print('Recall:      ',recall)
print('F1 score:    ',2*precision*recall/(precision+recall))

print("Sensitivity: ",sensitivity)
print("Specificity: ",specificity)

weights = LinearRegression().fit(x_train,y_train,sample_weight=(y_train*0.8+0.1))
y_pred = np.array(weights.predict(x_test))

print('\n**Measures for Linear Regression Classifier**')

tp,fp,fn,tn = confusion_matrix(y,(y_pred>=7)).ravel()
sensitivity = tp/(tp+fp)
specificity = tn/(fn+tn)
P = tp/(tp+fp)
R = tp/(tp+fn)
print('Accuracy:    ',(tp+tn)/(tp+tn+fp+fn))
print('Precision:   ',P)
print('Recall:      ',R)
print('F1 score:    ',(2*P*R)/(P+R))
print("Sensitivity: ",sensitivity)
print("Specificity: ",specificity)

svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm.fit(x_train, y_train)
y_s = svm.predict(x_test)

print('\n**Measures for Support Vector Machine**')

print('Accuracy:    ',accuracy_score(y,y_s))
print('Precision:   ',precision_score(y,y_s))
print('Recall:      ',recall_score(y,y_s))
print('F1 score:    ',f1_score(y,y_s))

tp,fp,fn,tn = confusion_matrix(y,y_s).ravel()
sensitivity = tp/(tp+fp)
specificity = tn/(fn+tn)
print("Sensitivity: ",sensitivity)
print("Specificity: ",specificity)

gnb = GaussianNB()
weight = (y_train*0.6+0.2)
naivebayes = gnb.fit(x_train, y_train,sample_weight = weight)
y_n = naivebayes.predict(x_test)

print('\n**Measures for Naive Bayesian**')

tp,fp,fn,tn = confusion_matrix(y,y_n).ravel()
sensitivity = tp/(tp+fp)
specificity = tn/(fn+tn)

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print('Accuracy:    ',accuracy_score(test[['quality']]>=7,naivebayes.predict(x_test)))
print('Precision:   ',precision)
print('Recall:      ',recall)
print('F1 score:    ',2*precision*recall/(precision+recall))

print("Sensitivity: ",sensitivity)
print("Specificity: ",specificity)
