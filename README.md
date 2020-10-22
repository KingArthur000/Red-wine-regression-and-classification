# Red-wine-regression-and-classification
## 1). Linear regression to detect/estimate the value of “quality” attributes of a wine in the dataset.
**Input:** Training dataset and testing dataset\
**Output:** Graphical representation of fitting a line over your training data and sum of square error calculation for test dataset.
*	The script imports few standard libraries to implement linear regression. They are Pandas, sklearn.linear_model, sklearn.metrics, sklearn.preprocessing and matplotlib.pyplot.
*	Pandas is used to import the data from the excel sheets, it is referenced using the reference name ‘pd’. A powerful library commonly implemented for data cleaning and data management.
*	From sklearn.linear_model we import the LinearRegression module, which is used to perform the linear regression task in this script.
*	From sklearn.metrics we import mean_squared_error to calculate the mean squared error of the model.
*	From sklearn.preprocessing we import normalize to normalize our data, which is a preprocessing technique implemented to derive better results. We convert the values of all the attributes into a particular interval so that the processing is made easy, by converting all the values into a particular interval.
*	matplotlib.pyplot is imported as ‘plt’, which is used to plot the results of the model graphically. It is commonly used library for data visualization.
*	Initially we import the data from the training and testing data sets. Here we use pd.read_csv(), which imports tabular data from the file mentioned between the paranthesis. For ex,
train = pd.read_csv("wineQualityRed_train.csv")
It stores the data in the excel file onto the dataframe variable train.
*	Then we print the dataframe using .frame() function. By default the function prints first five tuples, unless specified explicitly. For ex, 
print(train.head())
*	Then we split the data from the excel sheets into X and Y. We use simple dataframe column isolation techniques to perform this operation. We simply mention the column name to be stored in the variable inside the dataframe object. For ex,
y_train = train[['quality']]
X_train = train[['fixed acidity','volatile acidity','citric acid','residual sugar', 'chlorides', 'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
*	We then normalize the training and testing data. This is done by invoking the normalize function that is imported from the preprocessing library. We input the array containing unnormalized data and the output is the normalized data. For ex,
x_train = normalize(X_train)
*	Now the data is ready to be passed onto the model. Now we input the data to the linear regression model that is imported. Here we use the function LinearRegression(), with the defaut setting itself, the default values to each attributes are mentioned below ie, sklearn.linear_model.LinearRegression(*, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
The fit() attribute of this library is used to fit the model to the training data. It takes the X_train and y_train as the input. Here the sample_weights to the model is set to default. 
weights = LinearRegression().fit(x_train,y_train)
The output from the fit() attribute is the weights of the model, which when multiplied with the input tuples would give the predicted output. It is of the dimension (n*1), where n is the dimension of the data.
*	Then we use the .predict() function to predict the output for the test cases. It takes the x_test as input and output’s the y_pred. For ex,
y_pred = weights.predict(x_test)
*	The weights also has some other attributes that are useful, those are .coef_, .intercept_ to get the coefficient matrix (w) and the intercept (b). For ex,
weights.coef_, weights.intercept_
*	Now to get the mean squared error of the predicted output we use the function mean_squared_error(), where the inputs are the y_pred and the y_test, the output is the mean squared error itself.
mean_squared_error(y_test,y_pred)
*	From the coefficient matrix we could see the ‘sulphates’ column has the most weight, hence I have taken that column to fit the line in the visualization. Here three functions plot(), scatter(), show() are used to plot the line, data points and the output.
plt.plot(x_plot,weight_plot*x_plot+weights.intercept_-45)
plt.scatter(x_plot,y_plot,c = 'r',marker = 'o')
plt.show()

**The Output**
 
 <img src="/docs/Q_1_1.png">
 <img src="/docs/Q_1_2.png">
 
Hence the model mildly fits the data. The is around 0.45, which is pretty low. And the linear fit is also visualized. The code is present in Q_1.py file

## 2). Use the following classifier to detect the quality of wine. As all the attributes of the data sets have numerical values you can take “quality” attribute of the dataset as class label. In Quality attribute the values greater than or equal to 7 can be considered as “good “quality and the quality value less than 7 can be considered as “bad” quality. So now you have two class problem.
a.	Logistic Regression\
b.	Linear Regression as a classifier\
c.	SVM\
d.	Naïve Bayesian
### a. Logistic Regression classifier
*	The libraries used in the previous code is also used here, but it imports LogisticRegression from the sklearn.linear_model library, also numpy library is also imported.
*	The excel files are also read using the same pd.excel() function. But as this is a classification model, we normalize the Y from integer domain to the Boolean domain, by using simple numpy array comparison
y_train = np.array(train[['quality']]>=7)
*	Now the data is ready to be passed onto the model. Now we input the data to the logistic regression model that is imported. Here we use the function LogisticRegression(), with the defaut setting itself, the default values to each attributes are mentioned below ie, sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

The fit() attribute of this library is used to fit the model to the training data. It takes the X_train and y_train as the input. Here the sample_weights to the model is set to default. 
logistic = LogisticRegression(solver='saga',max_iter=10000).fit(x_train, y_train)
*	The score() function is used to determine the accuracy of the model, It takes the test inputs X and Y as the input and returns the output accuracy,
logistic.score(x_test,y_test)
*	To map the Boolean values to the {‘Good’,’Bad’}, we use np.where(). It takes the columned to be mapped and the final mapped value, output’s the mapped output
np.where(logistic.predict(x_test), 'Good', 'Bad')
*	Then we create subplots to output the good and bad wine counts of y_test and y_pred. We create two axes and allot each graph to each axis. For ex,
fig, (ax1, ax2) = plt.subplots(2)
test['class'].value_counts().plot(kind='bar',figsize=(10,8),title="Actual Quality of Red Wine",ax = ax1)

**The output**
 
 <img src="/docs/Q_2_1_1.png">
 <img src="/docs/Q_2_1_2.png">
 
The logistic classifier proves to be a pretty decent estimate for the classification problem and in general to the dataset. It has an accuracy of 89.4%. The model can be used as a classifier compared to the linear classifier that is discussed below, in terms of model construction.

### b. Linear Regression as a classifier
*	The libraries that were imported in the first question are also used here and a similar reading of input and model building was done, as it is the same linear regression model.
*	The regression was made into a classifier by converting the ‘quality’ attribute into Boolean values by using simple numpy comparison as done in the previous logistic regression classifier, as this is the same classification problem.

**The output**
 
 <img src="/docs/Q_2_2_1.png">
 <img src="/docs/Q_2_2_2.png">
 
The linear classifier proves to be a pretty bad estimate for the classification problem and in general to the dataset. It has an accuracy of 88.75%. Because even though the accuracy is high the model estimates ‘Bad’ for every tuple in the test set. Hence it is very bad estimate.

### c. Support Vector Machine 
*	The pandas, numpy and matplolib libraries are imported and used in the code as described in the previous classifiers. The new libraries imported are 
*	From sklearn.pipeline we import make_pipeline, to make a default pipeline for the dataset. 
*	From sklearn.preprocessing we import StandardScaler for preprocessing the data.
*	From sklearn.svm we import SVC for the make_pipeline function.
*	The data is taken from the excel files using the pd.excel() function and as discussed above it is stored in dataframe variables.
*	Initially the pipeline is made for the estimation using the make_pipeline function, it takes the input as the StandardScaler() and SVC() (with auto mode in gamma attribute). For ex,
svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
*	Then we fit the model using the fit() it takes the training data as the input and stores the weights in the svm variable, as stated below
svm.fit(x_train, y_train)
*	The remaining is the same as used in the above codes, and are used to plot the results.

**The output**
 
 <img src="/docs/Q_2_3_1.png">
 <img src="/docs/Q_2_3_2.png">

The SVM classifier proves to be a pretty good estimate for the classification problem and in general to the dataset. It has an accuracy of 88.5%. A much better model compared to the linear classifier in terms of model construction. 

### d. Naïve Bayesian
*	The pandas, numpy and matplolib libraries are imported and used in the code as described in the previous classifiers. The new libraries imported are 
*	From sklearn.naive_bayes we import GaussianNB, for performing the naives Bayesian classification.
*	The data is taken from the excel files using the pd.excel() function and as discussed above it is stored in dataframe variables.
*	We then train the model using the fit() function, it takes the X and Y and output’s the weights of the model, as stated below,
naivebayes = GaussianNB().fit(x_train, y_train)
*	The remaining is the same as used in the above codes, and are used to plot the results.

**The output**
 
 <img src="/docs/Q_2_4_1.png">
 <img src="/docs/Q_2_4_2.png">
 
The Naïve Bayes classifier proves to be a pretty good estimate for the classification problem and in general to the dataset. It has an accuracy of 85.4%. A much better model compared to the linear classifier in terms of model construction.




## 3). Compare all the classifier based on Accuracy, Precision, Recall, F-measure, Sensitivity, and Specificity and discuss the result
**Input:** Test results from task 2\
**Output:** values of each measures with respect to each classifiers
*	The classifier libraries are the same as the previous question, but are imported into the same code file. We import few other libraries for the measures calculation
*	From sklearn.metrics we import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix. For accuracy, precision, recall, F1, confusion matrix calculation respectively.
*	The input from excel files is taken using read_csv() and stored in dataframe variables. The models are also fitted to the datasets as described in the previous question. 
*	The measures are found using two methods, one using the above libraries and the other using confusion matrix.
*	For accuracy calculation, we use the accuracy_score function, as normalize and sample weights attributes are set to default
accuracy_score(y_test,y_pred)
*	For precision calculation, we use the precision_score function, as the following  attributes are set to default
sklearn.metrics.precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
precision_score(y_test,y_pred)
*	For recall calculation, we use the recall_score function, as the following attributes are set to default
sklearn.metrics.recall_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
recall_score(y_test,y_pred)
*	For F1 score calculation, we use the f1_score function, as the following attributes are set to default
sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
f1_score(y_test,y_pred)
*	For confusion matrix calculation, we use the confusion_matrix function, as the following attributes are set to default, the function output is the true positive, false positive, false negative, true negative values
sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)
confusion_matrix(y,(y_pred>=7)).ravel()
ravel() function is used to convert the 2-D matrix to a 1-D array
*	All the metrics can be found from the confusion matrix and the formulas are given below
Accuracy = (tp+tn)/(tp+tn+fp+fn)
Sensitivity = tp/(tp+fp)
Specificity = tn/(fn+tn)
Precision = tp/(tp+fp)
Recall = tp/(tp+fn)
F1 = (2*Precision*Recall)/(Precision+Recall)

**The output**

<img src="/docs/Q_3.png">
 
The Measures for the classifiers are attached above, from the results we could observe that even though linear classifier has a decent accuracy even though it predicted all the values to be Bad quality. So accuracy seems to be a bad measure for estimating a model. But seeing the precision logistic proves to be great classifier, on based on recall Naïve Bayesian is a great model. The naïve Bayesian model is also a great estimate when it comes to the F1 score, even though it is pretty bad according to modern estimators. Logistic is more sensitive and naïve Bayesian is more specific. So comparing performance across various models can prove to be a great advantage when selecting the model for actual usage.



## 4). Find the correlation between attributes and apply PCA (Discuss about the correlation with respect to wine dataset). While applying PCA leave attribute “quality” as it is. So you have total 11 attributes. All the classifiers (a to d) should be applied again over the newly constructed dataset (dataset constructed using PCA) for the following number of attributes and evaluated using measures given in (3)
a.	7 (i.e. from PCA we get 11 attribute so take first 7 attributes of the newly constructed dataset, call this dataset as redwine_7_training and redwine_7_testing)\
b.	4 (i.e. from PCA we get 11 attribute so take first 4 attributes of the newly constructed dataset, call this dataset as redwine_4_training and redwine_4_testing)\
**Input:** {redwine_7_training and redwine_7_testing, redwine_4_training and redwine_4_testing}\
**Output:** values of each measures with respect to each classifiers and each dataset (both redwine_7 and redwine_4)
So what do you observe when taking 7 attribute and when you are taking 4 attributes what impact does the dimensionality have in various classifiers (based on evaluation measures).
Note: While using PCA for construction of new data from the existing data, PCA should be applied over both training and testing data.
### a. Using 7 attributes from PCA
*	Apart from the libraries imported in the above question, we also import PCA from sklearn.decomposition and perform PCA and take the first seven most correlated attributes. The PCA is performed by the following codes,
pca = PCA(n_components=7)
x_train = pca.fit_transform(X_train)
x_test = pca.transform(X_test)
Here the PCA components are stored in the pca variable, and the weights are fit using the fit_transform() function and the same weights are used for the transform() function which in turn is used for the test dataset.
*	The remaining results are found using the same method as in the previous question.

**The output**

<img src="/docs/Q_4_1.png">
 
There is a decrease in accuracy for the logistic model and there is no difference in the linear model as the weights given to the classes are also not impacting the results. But for the remaining classifiers weights are been assigned, 0.8 for the Good class and 0.2 for the Bad class, as the dataset is hugely unbalanced. The accuracy has improved for the SVM and Naïve Bayesian model. So there is an advantage in removing least correlated variables from the training and testing set to get better derivation. There are also changes in other metrics due to the removal of the data from the dataset, as even a small data can also matter for the classification of the data.
### b. Using 4 attributes PCA
*	Apart from the libraries imported in the above question, we also import PCA from sklearn.decomposition and perform PCA and take the first four most correlated attributes. The PCA is performed by the following codes,
pca = PCA(n_components=4)
x_train = pca.fit_transform(X_train)
x_test = pca.transform(X_test)
Here the PCA components are stored in the pca variable, and the weights are fit using the fit_transform() function and the same weights are used for the transform() function which in turn is used for the test dataset.
*	The remaining results are found using the same method as in the previous question.

**The Output**

<img src="/docs/Q_4.png">
 
There is an increase in accuracy in the logistic model compared to the previous PC analysis, but there is a decrease in the accuracy in SVM and Naïve Bayesian models, this may be due to the removal of essential data, because the original dataset has 11 attributes but now it has only 4 attributes and this may lead to loss of essential classification data. But the linear model still has the same metrics due to more samples at the Bad quality and linear model couldn’t account for the unbalance in the dataset. The F1 scores of the SVM has decreased terribly and the Naïve Bayesian metrics have improved a lot. So here Naïve Bayesian proves to be a great estimate. All the classes are provided with necessary weights to account for the unbalance in the dataset.
