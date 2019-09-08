# -*- coding: utf-8 -*-
"""
features = ['gpa','grade_yr','program','sex','target']
[[ 24  45]
 [ 39 759]]

Recall Score:  0.347826086957
Spec  Score:  0.951127819549
"""
import pandas as pd
import ROC_Graph as g
import numpy as np

# Importing the dataset
dataset = pd.read_csv('2008-2018_Data_Unique_DiscreteV9.csv', dtype = object) # Training Data
#dataset = dataset[(dataset.grade_yr == '1.0') & (dataset.sex == 'F')]
features = ['sex','ethnicity','program','grade_yr','gpa','discipline','distance','target']
#features = ['gpa','grade_yr','program','sex','target']
dataset = dataset[features]

X = dataset.iloc[:, :-1]
X = pd.get_dummies(X, drop_first = True)
y = dataset.iloc[:, -1]
y = pd.get_dummies(y, drop_first = True)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""


# Fitting classifier to the Training set
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train.iloc[:,0])

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels = [1,0])
print(cm)
len(y_test[y_test == 'True'])

rc = cm[0,0] / (cm[0,0] + cm[0,1])
print('\nRecall Score: ', rc)
ppc = (cm[1,0] + cm[0,0]) / cm.sum()
print('Predicted Positive Condition: ', ppc)

prob = classifier.predict_proba(X_test)
prob = prob[:,1]

g.graph_roc(y_test, y_pred, prob, 
            'NB- Receiver Operating Characteristic', 
            '2008-2018 AUC = %0.3f', 
            '^', 'g')

'''
g.graph_PRCurve(y_test, y_pred, prob, 
            'NB - Precision vs Recall', 
            '2008-2018 AUC = %0.3f', 
            '^', 'g',
            len(y_test[y_test['True'] == 1.0]) / len(y_test))
'''
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

#Xn = sc.transform(Xn)

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
            '2019 AUC = %0.3f', 
            '^', 'b')
'''
g.graph_PRCurve(yn, y_predn, probn, 
            'NB - Precision vs Recall', 
            '2008-2018 AUC = %0.3f', 
            '^', 'g',
            len(y_test[y_test['True'] == 1.0]) / len(y_test))
'''

#############################
#                           #
#     Model Evaluation      #
#                           #
#############################

from sklearn.model_selection import cross_val_score

scores = cross_val_score(classifier, X_train, y_train.iloc[:,0], cv = 10, scoring = 'recall')
print(scores)
mean = scores.mean()
std = scores.std()
print('mean = ', mean)
print('STD = ',std)

#############################
#                           #
#     Get Predictions       #
#                           #
#############################
'''
test_set['prediction'] = y_predn
test_set['actual'] = yn
pred_set = test_set[['id', 'prediction', 'actual']]
pred_set = pred_set[pred_set.prediction == 1]

pred_set.to_excel('NB 2019 Turnover Predictions.xlsx', index = False)

# Extra EDA
eda_set = test_set[['id', 'sex', 'ethnicity', 'program','grade_yr',
                    'gpa', 'discipline', 'distance', 'target', 
                    'prediction', 'actual']]
eda_set = eda_set[eda_set.prediction == 1]
'''












