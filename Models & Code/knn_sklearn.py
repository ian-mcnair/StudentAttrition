"""
Implementation of K-NN Algorithm for Continuous Data
"""
import pandas as pd
import ROC_Graph as g

# Importing the dataset
dataset = pd.read_csv('2008-2018_Data_Unique_DiscreteV9.csv', dtype = object) # Training Data
features = ['sex','ethnicity','grade_yr','gpa','discipline','target']
features = ['gpa','grade_yr','program','sex','target']

dataset = dataset[features]
#Oversampling
#sample = dataset[dataset.target == 'True']
#dataset =pd.concat([dataset,sample,sample])

X = dataset.iloc[:, :-1]
X = pd.get_dummies(X, drop_first = True)
y = dataset.iloc[:, -1]
y = pd.get_dummies(y, drop_first = True)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)'''

# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
# Create your classifier here
classifier = KNeighborsClassifier(n_neighbors = 61,
                                  weights = 'distance', 
                                  metric = 'minkowski', 
                                  p = 1)
classifier.fit(X_train, y_train.iloc[:,0])
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
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
            'KNN - Receiver Operating Characteristic', 
            '2008-2018 AUC = %0.3f', 
            '^', 'g')
'''
g.graph_PRCurve(y_test, y_pred, prob, 
            'KNN - Precision vs Recall', 
            '2008-2018 AUC = %0.3f', 
            '^', 'g',
            len(y_test[y_test['True'] == 1.0]) / len(y_test))
'''
#############################
#                           #
#     Cross Validation      #
#                           #
#############################
'''
from sklearn.model_selection import GridSearchCV

parameters = {'n_neighbors': [7,11,21,31, 61]}

gs = GridSearchCV(estimator = classifier, 
                  param_grid = parameters,
                  scoring = 'recall',
                  cv = 10)

gs = gs.fit(X_train, y_train.iloc[:,0])
best_param = gs.best_params_
best_acc = gs.best_score_
print(best_param)
print(best_acc)
'''
'''
from sklearn.model_selection import cross_val_score

scores = cross_val_score(classifier, X_train, y_train.iloc[:,0], cv = 10, scoring = 'recall')
print(scores)
mean = scores.mean()
std = scores.std()
print('mean = ', mean)
print('STD = ',std)
'''
#############################
#                           #
#     Prediction Data       #
#                           #
#############################
test_set = pd.read_csv('2019_Data_Discrete.csv')
dataset = test_set[features]
Xn = dataset.iloc[:,:-1]
Xn = pd.get_dummies(Xn, drop_first = True)
Xn = Xn.reindex(columns = X.columns, fill_value=0)
yn = dataset.iloc[:,-1]
yn = pd.get_dummies(yn, drop_first = True)

#Xn = sc.transform(Xn)

y_predn = classifier.predict(Xn)
probn = classifier.predict_proba(Xn)
probn = probn[:,1]
cm = confusion_matrix(yn, y_predn, labels = [1,0])
print(cm)
rc = cm[0,0] / (cm[0,0] + cm[0,1])
print('\nRecall Score: ', rc)
spec = cm[1,1] / (cm[1,1] + cm[1,0])
print('Spec  Score: ', spec)


g.graph_roc(yn, y_predn, probn, 
            'KNN - Receiver Operating Characteristic', 
            '2019 AUC = %0.3f', 
            '^', 'b')
'''g.graph_PRCurve(yn, y_predn, probn, 
            'Knn - Precision vs Recall', 
            '2008-2018 AUC = %0.3f', 
            '^', 'b',
            len(y_test[y_test['True'] == 1.0]) / len(y_test))


test_set['prediction'] = y_predn
test_set['actual'] = yn
pred_set = test_set[['id', 'prediction', 'actual']]
pred_set = pred_set[pred_set.prediction == 1]

pred_set.to_excel('KNN 2019 Turnover Predictions.xlsx', index = False)
'''
#############################
#                           #
#     Model Evaluation      #
#                           #
#############################

from sklearn.model_selection import cross_val_score

scores = cross_val_score(classifier, X_train, y_train, cv = 10, scoring = 'recall')
print(scores)
mean = scores.mean()
std = scores.std()
print('mean = ', mean)
print('STD= ',std)
