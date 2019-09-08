# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:04:30 2019

@author: imcna
"""

import pandas as pd

vote = pd.read_excel('MajorityVote.xlsx')
y_test = vote.iloc[:,1]
y_pred = vote.iloc[:,0]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels = [1,0])
print(cm)
rc = cm[0,0] / (cm[0,0] + cm[0,1])
print('\nRecall Score: ', rc)
spec = cm[1,1] / (cm[1,1] + cm[1,0])
print('Spec  Score: ', spec)