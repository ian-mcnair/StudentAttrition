# -*- coding: utf-8 -*-
"""
This takes teh data and leaves it as continuous as possible.
Categorical Variables are still encoded using OneHotEncoding
"""

import pandas as pd
import ROC_Graph as g

# Importing the dataset
dataset = pd.read_csv('2008-2018_Data_NumericalV9.csv') # Training Data
#features = ['ethnicity','gpa','discipline','distance','target']
features = ['gpa','grade_yr','program','sex','target']
dataset = dataset[features]
X = dataset.iloc[:, 1:-1].astype(str)
X = pd.get_dummies(X, drop_first = True)
X = pd.concat([X, dataset[['gpa']]], axis = 1)
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


# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB, BernoulliNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels = [1,0])
print(cm)
rc = cm[0,0] / (cm[0,0] + cm[0,1])
print('\nRecall Score: ', rc)
spec = cm[1,1] / (cm[1,1] + cm[1,0])
print('Spec  Score: ', spec)

prob = classifier.predict_proba(X_test)
prob = prob[:,1]
g.graph_roc(y_test, y_pred, prob, 
            'NB - Receiver Operating Characteristic', 
            '2008-2018 AUC = %0.2f', 
            '^', 'g')
"""
g.graph_PRCurve(y_test, y_pred, prob, 
            'NB - Precision vs Recall', 
            '2008-2018 AUC = %0.2f', 
            '^', 'g',
            len(y_test[y_test['True'] == 1.0]) / len(y_test))
"""
#############################
#                           #
#     Prediction Data       #
#                           #
#############################
testset = pd.read_csv('2019_Data_Numerical.csv')
testset = testset[features]
Xn = testset.iloc[:, 1:-1].astype(str)
Xn = pd.get_dummies(Xn, drop_first = True)
Xn = pd.concat([Xn, testset[['gpa']]], axis = 1)
Xn = Xn.reindex(columns = X.columns, fill_value=0)
yn = testset.iloc[:,-1]
yn = pd.get_dummies(yn, drop_first = True)


y_predn = classifier.predict(Xn)
cm = confusion_matrix(yn, y_predn, labels = [1,0])
print(cm)
rc = cm[0,0] / (cm[0,0] + cm[0,1])
print('\nRecall Score: ', rc)
spec = cm[1,1] / (cm[1,1] + cm[1,0])
print('Spec  Score: ', spec)

probn = classifier.predict_proba(Xn)
probn = probn[:,1]
g.graph_roc(yn, y_predn, probn, 
            'NB - Receiver Operating Characteristic', 
            '2008-2018 AUC = %0.2f', 
            '1', 'b')