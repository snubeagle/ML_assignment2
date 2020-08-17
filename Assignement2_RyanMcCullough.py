#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:45:33 2020

@author: ryan
"""

import pandas as pd
from sklearn.model_selection import train_test_split as sklcv
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    pima = pd.read_csv("pima-indians-diabetes-database.csv", header=None, names=col_names)
    
    feature_cols = ['pregnant', 'glucose', 'bp', 'pedigree', 'age']
    y_cols = ['label']
    x = pima[feature_cols]
    y = pima[y_cols]
    
    xTrain, xTest, yTrain, yTest = sklcv(x, y, test_size=.4)
      
    model = LogisticRegression()
    
    model.fit(xTrain, yTrain.values.ravel())
    
    yPred = model.predict(xTest)
    
    confusionMTX = metrics.confusion_matrix(yTest, yPred)
    
    class_names = ['Genuine', 'Counterfeit']
    fig, ax = plt.subplots()
    sns.heatmap(confusionMTX, annot=True, cmap="YlGnBu", fmt='g', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    
    print("Precision: ", metrics.precision_score(yTest, yPred))
    print("Recall: ", metrics.recall_score(yTest, yPred))
    print("F score: ", metrics.f1_score(yTest, yPred))
    
    fpr, tpr, threshold = metrics.roc_curve(yTest, yPred)
    print("FPR: ",fpr)
    print("TPR: ",tpr)
    
    print("ROC AUC Score: ", metrics.roc_auc_score(yTest, yPred))
    roc_auc = metrics.roc_auc_score(yTest, yPred)
    
    plt.figure()
    y_pred_probabilities = model.predict_proba(xTest)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(yTest,  y_pred_probabilities)
    auc = metrics.roc_auc_score(yTest, y_pred_probabilities)
    plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
    plt.legend(loc=4)