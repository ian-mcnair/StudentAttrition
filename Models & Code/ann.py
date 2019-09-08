# -*- coding: utf-8 -*-
"""
features = ['gpa','ethnicity', 'grade_yr','target'] # 0.35 Thresh
[[ 29  40]
 [ 47 751]]

Recall Score:  0.420289855072
Spec  Score:  0.941102756892

features = ['gpa','grade_yr','program','sex','target'] #.45 Thresh
[[ 29  40]
 [ 49 749]]

Recall Score:  0.420289855072
Spec  Score:  0.938596491228
"""
import pandas as pd
import ROC_Graph as g
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('2008-2018_Data_Unique_DiscreteV9.csv', dtype = object) # Training Data
#dataset = dataset[dataset.grade_yr == '1.0']
#features = ['gpa','ethnicity', 'grade_yr','target'] # 0.25 Thresh
features = ['gpa','grade_yr','program','sex','target'] #.45 Thresh

#features = ['sex','ethnicity','program','grade_yr','gpa','discipline','distance','target']
dataset = dataset[features]
X = dataset.iloc[:, :-1]
X = pd.get_dummies(X, drop_first = True)
y = dataset.iloc[:, -1]
y = pd.get_dummies(y, drop_first = True)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
len(X.columns)
"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

# Importing the Keas Libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initializing the ANN
classifier = Sequential()

# Making the ANN
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(X.columns)))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['mae'] )

classifier.fit(X_train,y_train.iloc[:,0], batch_size = 32, epochs = 100)

# Predicting
threshold = .45
y_pred = classifier.predict(X_test)
y_predict = (y_pred > threshold)
cm = confusion_matrix(y_test,y_predict, labels = [1,0])
print(cm)
rc = cm[0,0] / (cm[0,0] + cm[0,1])
print('\nRecall Score: ', rc)
ppc = (cm[1,0] + cm[0,0]) / cm.sum()
print('Predicted Positive Condition: ', ppc)

prob = classifier.predict_proba(X_test)
g.graph_roc(y_test, y_predict, prob, 
            'ANN - Receiver Operating Characteristic', 
            '2008-2018 AUC = %0.3f', 
            '^', 'g')
"""
g.graph_PRCurve(y_test, y_pred, prob, 
            'ANN - Precision vs Recall', 
            '2008-2018 AUC = %0.3f', 
            '^', 'g',
            len(y_test[y_test['True'] == 1.0]) / len(y_test))
"""
# Predicting With Newest Set
test_set = pd.read_csv('2019_Data_Discrete.csv', dtype = object)
#test_set= test_set[test_set.grade_yr == '1']
dataset = test_set[features]
Xn = dataset.iloc[:,:-1]
Xn = pd.get_dummies(Xn, drop_first = True)
Xn = Xn.reindex(columns = X.columns, fill_value=0)
yn = dataset.iloc[:,-1]
yn = pd.get_dummies(yn, drop_first = True)
y_predn = classifier.predict(Xn)
y_predictn = (y_predn > threshold)
cm = confusion_matrix(yn, y_predictn, labels = [1,0])
print(cm)
rc = cm[0,0] / (cm[0,0] + cm[0,1])
print('\nRecall Score: ', rc)
ppc = (cm[1,0] + cm[0,0]) / cm.sum()
print('Predicted Positive Condition: ', ppc)

probn = classifier.predict_proba(Xn)
g.graph_roc(yn, y_predictn, probn, 
            'ANN - Receiver Operating Characteristic', 
            '2019 AUC = %0.3f', 
            '^', 'b')

#############################
#                           #
#     Get Predictions       #
#                           #
#############################
'''
test_set['prediction'] = y_predictn.astype(int)
test_set['actual'] = yn
pred_set = test_set[['id', 'prediction', 'actual']]
pred_set = pred_set[pred_set.prediction == 1]

pred_set.to_excel('ANN 2019 Turnover Predictions.xlsx', index = False)

# Extra EDA
eda_set = test_set[['id', 'sex', 'ethnicity', 'program','grade_yr',
                    'gpa', 'discipline', 'distance', 'target', 
                    'prediction', 'actual']]
eda_set = eda_set[eda_set.prediction == 1]

print(eda_set.religion.value_counts())
'''

# Evaluation of Model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(X.columns)))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['mae'] )
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 32, epochs = 100)
score = cross_val_score(estimator = classifier, X = X_train, y = y_train.iloc[:,0], cv = 10, scoring = 'recall')
mean = score.mean()
std = score.std()

print('mean = ', mean)
print('STD = ',std)
"""
'''
# Gridsearch
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units =15, init = 'uniform', activation = 'relu', input_dim = 30))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 15, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['mae'] )
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)

parameters = {'batch_size': [10,25,32], 
              'epochs': [50, 100],
              'optimizer': ['adam', 'rmsprop']}

gs = GridSearchCV(estimator = classifier, 
                  param_grid = parameters,
                  scoring = 'recall',
                  cv = 10)

gs = gs.fit(X_train, y_train)
best_param = gs.best_params_
best_acc = gs.best_score_
'''
