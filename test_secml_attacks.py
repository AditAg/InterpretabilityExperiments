import numpy as np
#import pandas as pd
import random
#import matplotlib.pyplot as plt
from sklearn import svm
#import secml
from secml.ml.features import CNormalizerMinMax
from secml.ml.classifiers.sklearn import c_classifier_sklearn
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.peval.metrics import CMetricAccuracy
from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.adv.attacks.evasion import CAttackEvasionPGD
from secml.adv.attacks.poisoning.c_attack_poisoning import CAttackPoisoning #This is an Abstract class
from secml.adv.attacks.poisoning.c_attack_poisoning_svm import CAttackPoisoningSVM #Only works on binary-classification SVM
from secml.ml.kernels import CKernelRBF
from secml.data import CDataset
from secml.array import CArray
from fns import load_sentiment_dataset

tr_X, tr_Y, cv_X, cv_Y, te_X, te_Y = load_sentiment_dataset(classification_type = 'binary')
all_classes = list(np.unique(te_Y))
print(all_classes)
tr_X, tr_Y = CArray(tr_X), CArray(tr_Y)
cv_X, cv_Y = CArray(cv_X), CArray(cv_Y)
te_X, te_Y = CArray(te_X), CArray(te_Y)

ds_tr_secml = CDataset(tr_X, tr_Y)
#print(ds_tr_secml.classes, ds_tr_secml.num_classes, ds_tr_secml.num_features, ds_tr_secml.num_samples)
ds_te_secml = CDataset(te_X, te_Y)
ds_cv_secml = CDataset(cv_X, cv_Y)

normalizer = CNormalizerMinMax()
ds_tr_secml.X = normalizer.fit_transform(ds_tr_secml.X)
ds_te_secml.X = normalizer.transform(ds_te_secml.X)
ds_cv_secml.X = normalizer.transform(ds_cv_secml.X)


# =============================================================================
# #TEST WITH SKLEARN SVM
# sklearn_clf = svm.SVC(C = 1, kernel = 'rbf', gamma = 1.0)
# secml_sklearn_clf = c_classifier_sklearn.CClassifierSkLearn(sklearn_clf)
# secml_sklearn_clf.fit(ds_tr_secml)
# preds = secml_sklearn_clf.predict(ds_te_secml.X)
# metric = CMetricAccuracy()
# acc = metric.performance_score(y_true = ds_te_secml.Y, y_pred = preds)
# print("Accuracy on test set: {:.2%}".format(acc))
# probs = secml_sklearn_clf.predict_proba(ds_te_secml.X)       #Doesn't work
# 
# #sklearn here isn't supported for performing adversarial attacks, only the native SVM of secml supports adversarial attacks
# ###############################################################
# 
# =============================================================================
x, y = ds_te_secml[:, :].X, ds_te_secml[:, :].Y     # This won't work if we want to specify the target
#class for each example

#secml_clf = CClassifierMulticlassOVA(CClassifierSVM, kernel = CKernelRBF(gamma = 10), C = 1)
secml_clf = CClassifierSVM(kernel = CKernelRBF(gamma = 10), C = 1)
secml_clf.fit(ds_tr_secml)
preds = secml_clf.predict(ds_te_secml.X)
metric = CMetricAccuracy()
acc = metric.performance_score(y_true = ds_te_secml.Y, y_pred = preds)
print("Accuracy on test set: {:.2%}".format(acc))

#Performing the attack
noise_type = 'l2'
dmax = 0.4
lb, ub = None, None # with 0, 1 it goes out of bounds
y_target = None #### Here y_target can be some class, indicating which class is expected for the adversarial example

#solver_params = {
#    'eta': 0.3,
#    'max_iter': 100,
#    'eps': 1e-4
#}
solver_params = {
    'eta': 0.05,
    'eta_min': 0.05,
    'eta_max': None,
    'max_iter': 100,
    'eps': 1e-6
}

# =============================================================================
# adv_ds_te_secml_X = CArray.empty(ds_te_secml.X.shape)
# adv_ds_te_secml_Y = CArray.empty(ds_te_secml.Y.shape)
# 
# 
# for i in range(ds_te_secml.X.shape[0]):
#     x, y = ds_te_secml[i, :].X, ds_te_secml[i, :].Y
#     # Run the evasion attack on x0
#     curr_class = y.item()
#     rem_classes = all_classes.copy()
#     rem_classes.remove(curr_class)
#     y_target = random.choice(rem_classes)
#     print(y_target)
#     #pgd_ls_attack = CAttackEvasionPGD(classifier = secml_clf, surrogate_classifier = secml_clf, surrogate_data = ds_tr_secml, distance=noise_type,
#     #dmax=dmax, lb=lb, ub=ub, solver_params=solver_params, y_target=y_target)
#     
#     poisoning_attack = CAttackPoisoningSVM(classifier = secml_clf, training_data = ds_tr_secml, surrogate_classifier = secml_clf, val = ds_cv_secml, 
#                                         surrogate_data = ds_tr_secml, distance = noise_type, dmax = 1, lb = ds_cv_secml.X.min(), ub = ds_cv_secml.X.max(), y_target = y_target, 
#                                         attack_classes = 'all', solver_params = solver_params)
#     #y_pred_pgdls, _, adv_ds_pgdls, _ = poisoning_attack.run(x, y, double_init=False)
#     y_pred_pgdls, _, adv_ds_pgdls, _ = poisoning_attack.run(x, y)
#     
#     print("Original x0 label: ", y.item())
#     print("Adversarial example label (PGD-LS): ", y_pred_pgdls.item())
#     
#     #print("Number of classifier gradient evaluations: {:}"
#     #      "".format(poisoning_attack.grad_eval))
#     adv_ds_te_secml_X[i, :] = adv_ds_pgdls.X
#     adv_ds_te_secml_Y[i] = y_pred_pgdls.item()
# 
# adv_ds_te_secml = CDataset(adv_ds_te_secml_X, adv_ds_te_secml_Y)
# =============================================================================


poisoning_attack = CAttackPoisoningSVM(classifier = secml_clf, training_data = ds_tr_secml, surrogate_classifier = secml_clf, val = ds_cv_secml, 
                                        surrogate_data = ds_tr_secml, distance = noise_type, dmax = 1, lb = ds_cv_secml.X.min(), ub = ds_cv_secml.X.max(), 
                                        attack_classes = 'all', solver_params = solver_params)
# Run the poisoning attack
print("Attack started...")
pois_y_pred, pois_scores, pois_ds, f_opt = poisoning_attack.run(ds_te_secml.X, ds_te_secml.Y)
print("Attack complete!")

# Evaluate the accuracy of the original classifier
acc = metric.performance_score(y_true=ds_te_secml.Y, y_pred=preds)
# Evaluate the accuracy after the poisoning attack
pois_acc = metric.performance_score(y_true=ds_te_secml.Y, y_pred=pois_y_pred)

print("Original accuracy on test set: {:.2%}".format(acc))
print("Accuracy after attack on test set: {:.2%}".format(pois_acc))


