import numpy as np
#import pandas as pd
import random
#import matplotlib.pyplot as plt
from sklearn import svm
#import secml
from secml.ml.features import CNormalizerMinMax
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.peval.metrics import CMetricAccuracy
from secml.ml.kernels import CKernelRBF
from cleverhans.attacks import CarliniWagnerL2, ProjectedGradientDescent, MomentumIterativeMethod, FastGradientMethod
from secml.adv.attacks import CAttackEvasionCleverhans
from collections import namedtuple
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


x0, y0 = ds_te_secml[0, :].X, ds_te_secml[0, :].Y     

secml_clf = CClassifierMulticlassOVA(CClassifierSVM, kernel = CKernelRBF(gamma = 10), C = 1)
secml_clf.fit(ds_tr_secml)
preds = secml_clf.predict(ds_te_secml.X)
metric = CMetricAccuracy()
acc = metric.performance_score(y_true = ds_te_secml.Y, y_pred = preds)
print("Accuracy on test set: {:.2%}".format(acc))


#Performing the attack
noise_type = 'l2'
dmax = 1
lb, ub = None, None # with 0, 1 it goes out of bounds
y_target = None #### Here y_target can be some class, indicating which class is expected for the adversarial example

x0, y0 = ds_te_secml[0, :].X, ds_te_secml[0, :].Y

print(y0.item())
if(y0.item() == 0):
   y_target = 1
else:
   y_target = 0

Attack = namedtuple('Attack', 'attack_cls short_name attack_params')
attacks = [Attack(FastGradientMethod, 'FGM', {'eps': dmax, 'clip_max': ub, 'clip_min' : lb, 'ord':2}),
Attack(ProjectedGradientDescent, 'PGD', {'eps': dmax, 'eps_iter': 0.05, 'nb_iter': 50, 'clip_max': ub, 'clip_min' : lb, 'ord':2, 'rand_init': False}),
Attack(MomentumIterativeMethod, 'MIM', {'eps': dmax, 'eps_iter': 0.05, 'nb_iter': 50, 'clip_max': ub, 'clip_min' : lb, 'ord':2, 'decay_factor': 1}),
Attack(CarliniWagnerL2, 'CW2', {'binary_search_steps': 1, 'initial_const': 0.2, 'confidence': 10, 'abort_early': True, 'clip_max': ub, 'clip_min' : lb, 'max_iterations':50, 'learning_rate': 0.1})]

for i, attack in enumerate(attacks):
   cleverhans_attack = CAttackEvasionCleverhans(classifier = secml_clf, surrogate_classifier = secml_clf, surrogate_data = ds_tr_secml, y_target = y_target, clvh_attack_class = attack.attack_cls, **attack.attack_params)
   print("Attack {:} started ..".format(attack.short_name))
   y_pred_CH, _, adv_ds_CH, _ = cleverhans_attack.run(x0, y0)
   print("Attack finished")
   print("Original x0 label : ", y0.item())
   print("Adversarial example label ({:}): "
   "".format(attack.attack_cls.__name__), y_pred_CH.item())
   print("Number of classifier function evaluations: {:}"
   "".format(cleverhans_attack.f_eval))
   print("Number of classifier gradient evaluations: {:}"
   "".format(cleverhans_attack.grad_eval))

