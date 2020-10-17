import numpy as np
from fns import load_original_sentiment_dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


tr_X, tr_Y, cv_X, cv_Y, te_X, te_Y = load_original_sentiment_dataset()
#For GaussianNB
#tuned_parameters = [{'var_smoothing':[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]}]
tuned_parameters = [{'alpha' : [100, 10, 5, 4, 3, 2, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}]
scores = ['precision']
for score in scores:
   print("# Tuning hyperparameters for %s" % score)
   clf = GridSearchCV(MultinomialNB(), tuned_parameters, scoring = '%s_macro' % score)
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


final_clf = MultinomialNB(parameters)
print("Hello")