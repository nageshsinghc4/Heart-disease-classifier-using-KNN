#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 17:48:21 2019

@author: nageshsinghchauhan
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from sklearn import metrics

data = pd.read_csv('/Users/nageshsinghchauhan/Downloads/ML/KNN_art/heart.csv')

#data exploration
#check
data.target.value_counts()
sns.countplot(x="target", data=data, palette="bwr")
plt.show()

#male vs female
data.sex.value_counts()

sns.countplot(x='sex', data=data, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()

#frequency of heart rate in Age category
pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()



#X -> matrix of independent variable
#y -> vector of dependent variable
X = data.iloc[:,:-1].values
y = data.iloc[:,13].values



#splitting the dataset into test and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.25, random_state= 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#K-NN classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifier = classifier.fit(X_train,y_train)
#prediction
y_pred = classifier.predict(X_test)
#check accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

