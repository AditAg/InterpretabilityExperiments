from sklearn import svm
import numpy as np
import math
from fns import load_sentiment_dataset

def compute_lppd(probs):
    #probs is an array of shape data_size X no_samples (sampled parameter set)
    no_sampled_parameters = probs.shape[1]
    data_size = probs.shape[0]
    sum_probs_per_observation = np.empty((data_size, 1))
    for i in range(data_size):
        probs_per_observation = probs[i]
        sum_probs_per_observation[i] = np.sum(probs_per_observation)
        sum_probs_per_observation[i] = (sum_probs_per_observation[i])/(no_sampled_parameters)
    
    log_vals = np.log(sum_probs_per_observation)
    #log_vals = np.log2(sum_probs_per_observation)
    return np.sum(log_vals)


def compute_penalty_term(probs):
    #probs is an array of shape data_size X no_samples (sampled parameter set)
    log_probs = np.log(probs)
    var_per_observation = np.var(log_probs, axis = 1)
    return np.sum(var_per_observation)

def calculate_waic(probs):
    #probs is an array of shape data_size X no_samples (sampled parameter set)
    lppd_val = compute_lppd(probs)
    penalty_term = compute_penalty_term(probs)
    return -2.0 * (lppd_val - penalty_term)

def get_classifiers_list(kernel):
    clfs = []
    if(kernel == 'linear'):
        parameters_list = [{'C' : 1}, {'C': 10}, {'C': 100}, {'C': 1000}]
        for params in parameters_list:
            dict_clf = {}
            new_clf = svm.SVC(C = params['C'], kernel = 'linear')
            dict_clf['clf'] = new_clf
            dict_clf['params'] = params
            clfs.append(dict_clf)
        
    elif(kernel == 'rbf'):
        parameters_list = [{'C' : 1, 'gamma': 'scale'}, {'C': 10, 'gamma': 'scale'}, {'C': 100, 'gamma': 'scale'}, {'C': 1000, 'gamma': 'scale'}, 
                           {'C' : 1, 'gamma': 'auto'}, {'C': 10, 'gamma': 'auto'}, {'C': 100, 'gamma': 'auto'}, {'C': 1000, 'gamma': 'auto'}]
        for params in parameters_list:
            dict_clf = {}
            new_clf = svm.SVC(C = params['C'], kernel = 'rbf', gamma = params['gamma'])
            dict_clf['clf'] = new_clf
            dict_clf['params'] = params
            clfs.append(dict_clf)
    
    elif(kernel == 'poly'):
        parameters_list = []
        C_vals = [1, 10, 100, 1000]
        gamma_vals = ['scale', 'auto']
        coef0_vals = [0.0, 1.0, 2.0, 3.0]
        degree_vals = [2]
        for C in C_vals:
            for gamma in gamma_vals:
                for coef0 in coef0_vals:
                    for degree in degree_vals:
                        parameters_list.append({'C': C, 'gamma' : gamma, 'coef0': coef0, 'degree' : degree})
                        
        for params in parameters_list:
            dict_clf = {}
            new_clf = svm.SVC(C = params['C'], kernel = 'poly', gamma = params['gamma'], coef0 = params['coef0'], degree = params['degree'])
            dict_clf['clf'] = new_clf
            dict_clf['params'] = params
            clfs.append(dict_clf)
    
    elif(kernel == 'sigmoid'):
        parameters_list = []
        C_vals = [1, 10, 100, 1000]
        gamma_vals = ['scale', 'auto']
        coef0_vals = [0.0, 1.0, 2.0, 3.0]
        for C in C_vals:
            for gamma in gamma_vals:
                for coef0 in coef0_vals:
                    parameters_list.append({'C': C, 'gamma' : gamma, 'coef0': coef0, 'degree' : degree})
                        
        for params in parameters_list:
            dict_clf = {}
            new_clf = svm.SVC(C = params['C'], kernel = 'isgmoid', gamma = params['gamma'], degree = params['degree'])
            dict_clf['clf'] = new_clf
            dict_clf['params'] = params
            clfs.append(dict_clf)
            
    return clfs

def K(X, xi, xj, params, kernel):
    if(kernel == 'linear'):
        return np.dot(xi, xj)
    
    if(kernel == 'rbf' or kernel == 'poly' or kernel == 'sigmoid'):
        if(params['gamma'] == 'scale'):
            gamma = 1/(xi.shape[0] * np.var(X))
        elif(params['gamma'] == 'auto'):
            gamma = 1/(xi.shape[0])
        
    if(kernel == 'rbf'):
        val = -gamma * ((np.linalg.norm(xi - xj))**2)
        return math.exp(val)
    elif(kernel == 'poly'):
        val = (np.dot(xi, xj) * gamma) + params['coef0']
        return math.pow(val, params['degree'])
    elif(kernel == 'sigmoid'):
        val = (np.dot(xi, xj) * gamma) + params['coef0']
        return np.tanh(val)
    else:
        print("Invalid")
        return None
    
def dist_separating_hyperplane(X, Y, i, support_vectors_indexes, params, kernel):
    xi, yi = X[i], Y[i]
    sum_val = 0.0
    for i in range(support_vectors_indexes.shape[0]):
        sv = X[support_vectors_indexes[i]]
        sv_Y = Y[support_vectors_indexes[i]]
        k_val = K(X, xi, sv, params, kernel)
        sum_val += (sv_Y * params['C'] * k_val)
    
    if(yi == 0):
        return -1*sum_val
    else:
        return yi*sum_val

def l(d):
    return max(1-d, 0)

def compute_probs(kernel, data_X, data_Y):
    clf_list = get_classifiers_list(kernel)
    no_sampled_parameters = len(clf_list)
    
    log_probs = np.empty((no_sampled_parameters, data_X.shape[0]))
    index = 0
    for classifier in clf_list:
        clf_ = classifier['clf']
        clf_.fit(data_X, data_Y)
        support_vectors_indexes = clf_.support_
        for i in range(data_X.shape[0]):
            dist_sep_plane = dist_separating_hyperplane(data_X, data_Y, i, support_vectors_indexes, clf_['params'], kernel)
            l_dist = l(dist_sep_plane)
            #above l_dist = -p(x,y|w), so we have to take its negative
            log_probs[i][index] = -1.0 * l_dist
        index = index + 1
    return log_probs


tr_X, tr_Y, cv_X, cv_Y, te_X, te_Y = load_sentiment_dataset()

log_probs = compute_probs('rbf', tr_X, tr_Y)
waic = calculate_waic(log_probs)

print(waic)