# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 23:27:57 2019

@author: imcna
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
import ROC_Graph as g

# Importing the dataset
dataset = pd.read_csv('2008-2018_Data_NumericalV9.csv') # Training Data
features = ['gpa','discipline','distance','target']
#features = ['gpa','grade_yr','program','sex','target']
dataset = dataset[features]
X = dataset.iloc[:,:-1]
X = pd.get_dummies(X, drop_first = True)
X = pd.concat([X, dataset[['gpa','discipline','distance']]], axis = 1)
y = dataset.iloc[:, -1]
y = pd.get_dummies(y, drop_first = True)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the Keas Libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initializing the ANN
classifier = Sequential()

# Making the ANN
classifier.add(Dense(units = 3, init = 'uniform', activation = 'sigmoid', input_dim = len(X.columns)))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 2, init = 'uniform', activation = 'sigmoid'))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['mae'] )

classifier.fit(X_train,y_train, batch_size = 32, epochs = 100)

# Predicting
threshold = 0.025
y_pred = classifier.predict(X_test)
y_predict = (y_pred > threshold)
cm = confusion_matrix(y_test,y_predict, labels = [1,0])
print(cm)
rc = cm[0,0] / (cm[0,0] + cm[0,1])
print('\nRecall Score: ', rc)
spec = cm[1,1] / (cm[1,1] + cm[1,0])
print('Spec  Score: ', spec)

# ROC
prob = classifier.predict_proba(X_test)
g.graph_roc(y_test, y_predict, prob, 
            'ANN - Receiver Operating Characteristic', 
            '2008-2018 AUC = %0.2f', 
            '^', 'g')

# Predicting With Newest Set
testset = pd.read_csv('2019_Data_Numerical.csv')
testset = testset[features]
Xn = testset.iloc[:, :-1]
Xn = pd.get_dummies(Xn, drop_first = True)
Xn = pd.concat([Xn, testset[['gpa','discipline','distance']]], axis = 1)
Xn = Xn.reindex(columns = X.columns, fill_value=0)
yn = testset.iloc[:,-1]
yn = pd.get_dummies(yn, drop_first = True)

y_predn = classifier.predict(Xn)
y_predictn = (y_predn > threshold)
cm = confusion_matrix(yn, y_predictn, labels = [1,0])
print(cm)
rc = cm[0,0] / (cm[0,0] + cm[0,1])
print('\nRecall Score: ', rc)
spec = cm[1,1] / (cm[1,1] + cm[1,0])
print('Spec  Score: ', spec)

probn = classifier.predict_proba(Xn)
g.graph_roc(yn, y_predictn, probn, 
            'ANN - Receiver Operating Characteristic', 
            '2008-2018 AUC = %0.2f', 
            '^', 'b')