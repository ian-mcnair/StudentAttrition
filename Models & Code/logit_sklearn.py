# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:56:01 2019

@author: imcna
"""
import pandas as pd
import ROC_Graph as g

# Importing the dataset
dataset = pd.read_csv('2008-2018_Data_Unique_DiscreteV9.csv', dtype = object) # Training Data
#dataset = dataset[(dataset.grade_yr == '1.0') & (dataset.sex == 'F')]
features = ['sex','ethnicity','program','grade_yr','gpa','discipline','distance','target']
features = ['gpa','grade_yr','program','sex','target']
dataset = dataset[features]
X = dataset.iloc[:, :-1]
X = pd.get_dummies(X, drop_first = True)
y = dataset.iloc[:, -1]
y = pd.get_dummies(y, drop_first = True)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels = [1,0])
print(cm)
rc = cm[0,0] / (cm[0,0] + cm[0,1])
print('\nRecall Score: ', rc)
ppc = (cm[1,0] + cm[0,0]) / cm.sum()
print('Predicted Positive Condition: ', ppc)


prob = classifier.predict_proba(X_test)
prob = prob[:,1]
g.graph_roc(y_test, y_pred, prob, 
            'NB - Receiver Operating Characteristic', 
            '2008-2018 AUC = %0.2f', 
            '^', 'g')
#############################
#                           #
#     Prediction Data       #
#                           #
#############################

test_set = pd.read_csv('2019_Data_Discrete.csv', dtype = object)
#test_set = test_set[test_set.grade_yr == '1']
dataset = test_set[features]
Xn = dataset.iloc[:,:-1]
Xn = pd.get_dummies(Xn, drop_first = True)
Xn = Xn.reindex(columns = X.columns, fill_value=0)
yn = dataset.iloc[:,-1]
yn = pd.get_dummies(yn, drop_first = True)

y_predn = classifier.predict(Xn)
cm = confusion_matrix(yn, y_predn, labels = [1,0])
print(cm)
rc = cm[0,0] / (cm[0,0] + cm[0,1])
print('\nRecall Score: ', rc)
ppc = (cm[1,0] + cm[0,0]) / cm.sum()
print('Predicted Positive Condition: ', ppc)

probn = classifier.predict_proba(Xn)
probn = probn[:,1]
g.graph_roc(yn, y_predn, probn, 
            'NB - Receiver Operating Characteristic', 
            '2008-2018 AUC = %0.2f', 
            '^', 'b')