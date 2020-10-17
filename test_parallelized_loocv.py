#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:25:29 2020

@author: adit
"""


from fns import load_sentiment_dataset
from sklearn.model_selection import KFold
import itertools
import multiprocessing as mp
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
from sklearn import svm
import pandas as pd

results = []
def log_result(x):
    results.append(x)
    
    
def benchmark_models(X, y, split):
    """
    Helper function to benchmark models
    X : array
    y : array
    split : tuple
     Training and test indices (train_idx, test_idx)
    """
    X_train, y_train = X[split[0],:], y[split[0]]
    X_test, y_test   = X[split[1],:], y[split[1]]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    labels = [0, 1]
    
    model_library = {}
    # One candidate model
    model_library["svm"] = svm.SVC(C = 1, kernel = 'linear', probability = True)

    results = {}
    for model_name, model in model_library.items():
        # Train the model
        model.fit(X_train, y_train)
        # Make predictions on the test data
        pred_test = model.predict(X_test)
        #pred_test = model.predict_proba(X_test)[:,1]
        # Evaluate the model
        #results[model_name] = roc_auc_score(y_test, pred_test) ###### roc_auc_score requires multiple labels, whereas when doing LOOCV, there would be only one label class in pred_test
        results[model_name] = {}
        results[model_name]['accuracy'] = accuracy_score(y_test, pred_test)
        #results[model_name]['f1_score'] = f1_score(y_test, pred_test, labels = labels)
        results[model_name]['log_loss'] = log_loss(y_test, pred_test, labels = labels)
    
    print(results)
    return pd.DataFrame(results)


tr_X, tr_Y, cv_X, cv_Y, te_X, te_Y = load_sentiment_dataset()
n_folds = tr_X.shape[0]
kf = KFold(n_splits = n_folds, shuffle = True, random_state = None)

cv_index = [(i, j) for i, j in kf.split(tr_X)]
pool = mp.Pool(mp.cpu_count() - 3)
#benchmark_models(tr_X, tr_Y, split = cv_index[0])
for fold in cv_index:
    pool.apply_async(benchmark_models, args = (tr_X, tr_Y, fold), callback = log_result)
    
pool.close()
pool.join()
#result = pd.concat(results, axis = 0, sort = True)
#result.index.name = "metric"
#result.reset_index()
#average = result.groupby(['metric']).mean()
result = pd.concat(results, axis = 1)
average = result.mean(axis = 1)
print(average)
