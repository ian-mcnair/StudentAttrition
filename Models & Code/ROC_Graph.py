# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:41:31 2019

@author: imcna
"""

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def graph_roc(y_test, y_pred, prob, title, label, marker, color):
    fpr, tpr, thresholds = roc_curve(y_test, prob)
    auc_score = roc_auc_score(y_test, prob)
    plt.title(title)
    plt.plot(fpr, tpr, 
             label = label % auc_score, 
             marker = marker,
             color = color)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def graph_PRCurve(y_test, y_pred, prob, title, label, marker, color, base = 0.5):
    precision, recall, thresholds = precision_recall_curve(y_test, prob)
    auc_score = auc(recall, precision)
    plt.title(title)
    plt.plot(recall, precision,
             label = label % auc_score, 
             marker = marker,
             color = color)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [base, base],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
