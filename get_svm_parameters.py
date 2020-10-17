# =============================================================================
# import numpy as np
# from fns import load_sentiment_dataset
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.model_selection import GridSearchCV
# 
# 
# tr_X, tr_Y, cv_X, cv_Y, te_X, te_Y = load_sentiment_dataset()
# cf_tr_X, cf_tr_Y, cf_cv_X, cf_cv_Y, cf_te_X, cf_te_Y = load_sentiment_dataset(classification_type = 'multi')
# #tuned_parameters = [{'kernel':['rbf'], 'gamma':[1e-3, 1e-4], 'C' : [1, 10, 100, 1000]}, {'kernel' : ['linear'], 'C' : [1, 10, 100, 1000]}]
# #tuned_parameters = [{'kernel':['rbf'], 'gamma':[1e3, 1e2, 10, 5, 3, 2, 1, 1e-1, 1e-2, 1e-3, 1e-4], 'C' : [0.1, 1, 10, 100, 1000]}]
# tuned_parameters = [{'kernel':['poly'], 'gamma':[10, 1, 1e-1], 'C' : [0.1, 1, 10], 'coef0' : [0.0, 2.0], 'degree' : [2, 3]}]
# 
# # =============================================================================
# scores = ['precision']
# for score in scores:
#     print("# Tuning hyperparameters for %s" % score)
#     clf = GridSearchCV(SVC(), tuned_parameters, scoring = '%s_macro' % score)
#     clf.fit(tr_X, tr_Y)
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r" % (mean, std*2, params))
#     print()
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = te_Y, clf.predict(te_X)
#     print(classification_report(y_true, y_pred))
#     print()
#     parameters = clf.best_params_
# # =============================================================================
# # =============================================================================
# # final_cf = SVC(C = 1, kernel = 'linear')
# # final_cf.fit(tr_X, tr_Y)
# # y_true, y_pred = cf_te_Y, final_cf.predict(cf_te_X)
# # print(accuracy_score(y_true, y_pred))
# # print(classification_report(y_true, y_pred))
# # print(accuracy_score(te_Y, final_cf.predict(te_X)))
# # =============================================================================
# =============================================================================

import numpy as np
from load_combined_dataset import load_sentiment_dataset
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV


tr_X, tr_Y, cv_X, cv_Y, te_X, te_Y = load_sentiment_dataset()
#tuned_parameters = [{'kernel':['rbf'], 'gamma':[1e-3, 1e-4], 'C' : [1, 10, 100, 1000]}, {'kernel' : ['linear'], 'C' : [1, 10, 100, 1000]}]
tuned_parameters = [{'kernel':['rbf'], 'gamma':[1e3, 1e2, 10, 5, 3, 2, 1, 1e-1, 1e-2, 1e-3, 1e-4], 'C' : [0.1, 1, 10, 100, 1000]}]
#tuned_parameters = [{'kernel':['poly'], 'gamma':[10, 1, 1e-1], 'C' : [0.1, 1, 10], 'coef0' : [0.0, 2.0], 'degree' : [2, 3]}]

# =============================================================================
scores = ['precision']
for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    clf = GridSearchCV(SVC(), tuned_parameters, scoring = '%s_macro' % score)
    clf.fit(tr_X, tr_Y)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std*2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = te_Y, clf.predict(te_X)
    print(classification_report(y_true, y_pred))
    print()
    parameters = clf.best_params_
# =============================================================================
# =============================================================================
final_cf = SVC(**clf.best_params_)
print(final_cf.get_params())
final_cf.fit(tr_X, tr_Y)
y_true, y_pred = te_Y, final_cf.predict(te_X)
print("WAIC selected :" , accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))

interpret_clf = SVC(C= 1, kernel = 'linear')
interpret_clf.fit(tr_X, tr_Y)
y_true, y_pred = te_Y, interpret_clf.predict(te_X)
print("Interpretability selected :" , accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
