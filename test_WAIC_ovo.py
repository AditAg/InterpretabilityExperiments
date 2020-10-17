from sklearn import svm
import numpy as np
import math
#from scipy.special import beta, betainc
from fns import load_sentiment_dataset

###########################################CARS DATASET: 398 obs. of  11 variables:

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

def calculate_waic(probs, class_array, compute_per_class):
    #probs is an array of shape data_size X no_samples (sampled parameter set)
    if(compute_per_class == True):
        waic_vals = []
        classes_ = np.unique(class_array)
        for class_ in classes_:
            indices = (class_array == class_).nonzero()
            probs_per_class = probs[indices]
            no_queries_per_class = probs_per_class.shape[0]
            lppd_val = compute_lppd(probs_per_class)
            penalty_term = compute_penalty_term(probs_per_class)
            waic_val = -2.0*(lppd_val - penalty_term)
            waic_vals.append((waic_val/no_queries_per_class))
            print("Class: ", class_, ", No_samples: ", no_queries_per_class)
        return waic_vals
        
    else:    
        lppd_val = compute_lppd(probs)
        penalty_term = compute_penalty_term(probs)
        return -2.0 * (lppd_val - penalty_term)

def get_dict_clf(kernel, clf, params):
    dict_clf = {}
    dict_clf['clf'] = clf
    dict_clf['params'] = params
    return dict_clf
    
def get_classifiers_list(kernel, data_X, data_Y, calc_data_X, calc_data_Y, test_X, test_Y):
    clfs = []
    if(kernel == 'linear'):
        parameters_list = [{'C' : 0.1}, {'C': 10}, {'C': 100}, {'C': 1000}]
        #parameters_list = [{'C' : 0.1}]
        for params in parameters_list:
            new_clf = svm.SVC(C = params['C'], kernel = 'linear', decision_function_shape = 'ovo')
            dict_clf = get_dict_clf(kernel, new_clf, params)
            clfs.append(dict_clf)
        
    elif(kernel == 'rbf'):
        #########gamma = scale, autp works only in the latest version of scikit learn -> 0.20
        #parameters_list = [{'C' : 1, 'gamma': 'scale'}, {'C': 10, 'gamma': 'scale'}, {'C': 100, 'gamma': 'scale'}, {'C': 1000, 'gamma': 'scale'}, 
        #                   {'C' : 1, 'gamma': 'auto'}, {'C': 10, 'gamma': 'auto'}, {'C': 100, 'gamma': 'auto'}, {'C': 1000, 'gamma': 'auto'}]
        #parameters_list = [{'C' : 1, 'gamma': 3}, {'C' : 1, 'gamma': 2}, {'C' : 1, 'gamma': 1}, {'C' : 1, 'gamma': 0.1},
        #                   {'C': 10, 'gamma': 2}, {'C': 10, 'gamma': 1}, {'C': 10, 'gamma': 0.1}, {'C': 10, 'gamma': 0.01},
        #                   {'C': 100, 'gamma': 2}, {'C': 100, 'gamma': 1}, {'C': 100, 'gamma': 0.1}, {'C': 100, 'gamma': 0.01}, 
        #                   {'C': 1000, 'gamma': 5}, {'C': 1000, 'gamma': 1}, {'C': 1000, 'gamma': 0.1}, {'C': 1000, 'gamma': 0.01}, {'C': 1000, 'gamma': 0.001}]
        parameters_list = [{'C' : 1, 'gamma': 1}, {'C' : 1, 'gamma': 0.1},
                           {'C': 10, 'gamma': 0.1}, {'C': 10, 'gamma': 0.01},
                           {'C': 100, 'gamma': 1}, {'C': 100, 'gamma': 0.1}, {'C': 100, 'gamma': 0.01}, 
                           {'C': 1000, 'gamma': 5}, {'C': 1000, 'gamma': 1}, {'C': 1000, 'gamma': 0.1}]
        
        #parameters_list = [{'C' : 1, 'gamma': 3}, {'C' : 10, 'gamma': 0.1}]
        for params in parameters_list:
            new_clf = svm.SVC(C = params['C'], kernel = 'rbf', gamma = params['gamma'], decision_function_shape = 'ovo')
            #new_clf.fit(data_X, data_Y)
            dict_clf = get_dict_clf(kernel, new_clf, params)
            clfs.append(dict_clf)
    
    elif(kernel == 'poly'):
        #parameters_list = []
        #C_vals = [0.1, 1, 10, 100]
        #gamma_vals = [1e-3, 1e-4]
        #gamma_vals = ['scale', 'auto']
        #coef0_vals = [0.0, 1.0, 2.0, 3.0]
        #degree_vals = [2, 3, 4]
        #for C in C_vals:
        #    for gamma in gamma_vals:
        #        for coef0 in coef0_vals:
        #            for degree in degree_vals:
        #                parameters_list.append({'C': C, 'gamma' : gamma, 'coef0': coef0, 'degree' : degree})
        
        parameters_list = [{'C' : 1, 'gamma': 1, 'coef0' : 0.0, 'degree' : 3}, {'C' : 1, 'gamma': 0.1, 'coef0' : 1.0, 'degree' : 2},
                           {'C': 10, 'gamma': 0.1, 'coef0' : 1.0, 'degree' : 2}, {'C': 10, 'gamma': 0.01, 'coef0' : 0.0, 'degree' : 3},
                           {'C': 100, 'gamma': 0.1, 'coef0' : 0.0, 'degree' : 2}, {'C': 100, 'gamma': 0.01, 'coef0' : 2.0, 'degree' : 3}]
        #, {'C': 100, 'gamma': 0.1, 'coef0' : 1.0, 'degree' : 2}, {'C': 100, 'gamma': 0.1, 'coef0' : 2.0, 'degree' : 3}, {'C': 1, 'gamma': 0.1, 'coef0' : 2.0, 'degree' : 3}, {'C': 10, 'gamma': 0.1, 'coef0' : 2.0, 'degree' : 3}]                
        for params in parameters_list:
            new_clf = svm.SVC(C = params['C'], kernel = 'poly', gamma =  params['gamma'], coef0 = params['coef0'], degree = params['degree'], decision_function_shape = 'ovo')
            dict_clf = get_dict_clf(kernel, new_clf, params)
            clfs.append(dict_clf)
    
    elif(kernel == 'sigmoid'):
        parameters_list = []
        C_vals = [1, 10, 100, 1000]
        gamma_vals = [1e-3, 1e-4]
        #gamma_vals = ['scale', 'auto']
        coef0_vals = [0.0, 1.0, 2.0, 3.0]
        for C in C_vals:
            for gamma in gamma_vals:
                for coef0 in coef0_vals:
                    parameters_list.append({'C': C, 'gamma' : gamma, 'coef0': coef0})
                        
        for params in parameters_list:
            new_clf = svm.SVC(C = params['C'], kernel = 'sigmoid', gamma = params['gamma'], coef0 = params['coef0'], degree = params['degree'], decision_function_shape = 'ovo')
            dict_clf = get_dict_clf(kernel, new_clf, params)
            clfs.append(dict_clf)
    else:
        print("Bad kernel")
        return None
    
    new_data_X = kernel_function(kernel, data_X, params)
    new_calc_data_X = kernel_function(kernel, calc_data_X, params)
    new_test_X = kernel_function(kernel, test_X, params)
    dict_clfs = {}
    dict_clfs['clf_list'] = clfs
    dict_clfs['tr_data_X'] = new_data_X
    dict_clfs['tr_data_Y'] = data_Y
    dict_clfs['calc_data_X'] = new_calc_data_X
    dict_clfs['calc_data_Y'] = calc_data_Y
    dict_clfs['test_X'] = new_test_X
    dict_clfs['test_Y'] = test_Y
        
    return dict_clfs

def fact(n):
    res = 1
    for i in range(2, n+1):
        res = res * i
    return res

def ncr(n, r):
    return (fact(n)/(fact(r) * fact(n-r)))

def get_new_no_features(shape, degree):
    if(degree == 2):
        single_terms = shape + 1
        double_terms = ncr(shape + 1, 2)
        return single_terms + double_terms
    else:
        return shape + 1
        

def kernel_function(fn, x, params):
    return x           

#h should be a positive integrable function such that h(x1) = h(x2) for ||x1|| = ||x2||
#Examples given in paper: h(x) = exp(-<x, c1Ex>), h(x) = c2[||x||< c3] where c1, c2, c3 are arbitrary strictly positive scalars and E is the identity matrix
def h_general(x, params, kernel, X = None):
    c1 = 2.0
    outp = c1 * K_func(x, x, params, kernel, X)
    return math.exp(-1.0 * outp)

def K_func(xi, xj, params, kernel, X = None):
    if(kernel == 'linear'):
        return np.dot(xi, xj)
    
    if(kernel == 'rbf' or kernel == 'poly' or kernel == 'sigmoid'):
        if(params['gamma'] == 'scale'):
            gamma = 1/(xi.shape[0] * np.var(X))
        elif(params['gamma'] == 'auto'):
            gamma = 1/(xi.shape[0])
        else:
            gamma = params['gamma']
        
    if(kernel == 'rbf'):
        val = gamma * ((np.linalg.norm(xi - xj))**2)
        return math.exp(-1.0*val)
    elif(kernel == 'poly'):
        val = (np.dot(xi, xj) * gamma) + params['coef0']
        return math.pow(val, params['degree'])
    elif(kernel == 'sigmoid'):
        val = (np.dot(xi, xj) * gamma) + params['coef0']
        return np.tanh(val)
    else:
        print("Invalid")
        return None


def dist_separating_hyperplane(x, weights, y, b):
    if(y == 0):
        y = -1
    return (np.dot(weights, x) + b) * y

def dist_separating_hyperplane_general(X, params, kernel, x, y, coefs, support_vectors, intercepts):
    assert(coefs.shape[1] == len(support_vectors))
    if(y == 0):
        y = -1
    distances = []
    for class_ in range(coefs.shape[0]):
        dot_product = 0
        for i in range(coefs.shape[1]):
            val = K_func(x, support_vectors[i], params, kernel, X)
            val = val * coefs[class_][i]
            dot_product = dot_product + val
        dot_product = dot_product + intercepts[class_]
        distances.append(abs(y*dot_product))
    return min(distances)                           #TO VERIFY IF CORRECT OR NOT


#THIS IS INCORRECT BECAUSE I HAVE TO TRANSFORM THE SUPPORT_VECTORS USING THE PHI FUNCTION AND THEN DO THE MULTIPLICATION.
def get_weights(coefs, support_vectors):
    weights = np.empty(coefs.shape + (support_vectors[0].shape[0], ))
    for i in range(coefs.shape[0]):
        for j in range(coefs.shape[1]):
            weights[i][j] = coefs[i][j]*support_vectors[j]
    return weights
    
def l_func(d):
    return max(1-d, 0)

#Z has to be used as a normalization constant.
#Based on paper, Z = 1/(integration over all x of the h_val*e^(-l_dist) values.) See paper for clarity.
def Z(weights):
    norm_val = np.linalg.norm(weights)
    return 1/(norm_val + 1e-9)
    
def compute_probs(kernel, dict_clfs, tr_samples, te_samples):
    no_sampled_parameters = len(dict_clfs['clf_list'])
    
    train_probs = np.empty((tr_samples, no_sampled_parameters))
    test_probs = np.empty((te_samples, no_sampled_parameters))
    train_class = np.empty((tr_samples, ))
    test_class = np.empty((te_samples, ))
    index = 0
    
    tr_data_X = dict_clfs['tr_data_X']
    tr_data_Y = dict_clfs['tr_data_Y']
    calc_data_X = dict_clfs['calc_data_X']
    calc_data_Y = dict_clfs['calc_data_Y']
    test_X = dict_clfs['test_X']
    test_Y = dict_clfs['test_Y']
    
    for classifier in dict_clfs['clf_list']:
        clf = classifier['clf']
        params = classifier['params']
        print (params)
        print(tr_data_X.shape, tr_data_Y.shape)
        clf.fit(tr_data_X, tr_data_Y)
        #for linear kernel and binary classification
        if(kernel == 'linear' and np.unique(tr_data_Y).shape[0] == 2):
            #Works for 2-class classification only
            w = clf.coef_[0]
            b = clf.intercept_[0]
            #data transformed to higher dimensions, here =x as it is linear kernel
            sum_probs = 0.0
            print("Train data")
            for i in range(tr_samples):
                dist_sep_plane = dist_separating_hyperplane(calc_data_X[i], w, calc_data_Y[i], b)
                l_dist = l_func(dist_sep_plane)
                #phi(x) = x
                h_val = h_general(calc_data_X[i], params, kernel)
                z_val = Z(w)
                total_prob = math.exp(-1.0*l_dist) * h_val * z_val
                sum_probs = sum_probs + total_prob
                #print(total_prob)
                train_probs[i][index] = total_prob
                train_class[i] = calc_data_Y[i]
            #Actual normalization is being done here. Approximate the normalization term based on the current dataset, as integration over all possible (x,y) pairs is not tractable.  
            train_probs[:, index] = train_probs[:, index]/(sum_probs)
            sum_probs = 0.0
            print("Test data")
            for i in range(te_samples):
                dist_sep_plane = dist_separating_hyperplane(test_X[i], w, test_Y[i], b)
                l_dist = l_func(dist_sep_plane)
                #phi(x) = x
                h_val = h_general(test_X[i], params, kernel)
                z_val = Z(w)
                total_prob = math.exp(-1.0*l_dist) * h_val * z_val
                sum_probs = sum_probs + total_prob
                #print(total_prob)
                test_probs[i][index] = total_prob
                test_class[i] = test_Y[i]
            test_probs[:, index] = test_probs[:, index]/(sum_probs)
            
            index = index + 1
        
        elif(kernel == 'rbf' or kernel == 'poly' or kernel == 'sigmoid' or kernel == 'linear'):
            sv_indices = clf.support_
            
            dict_svs = {}
            for index1 in sv_indices:
                class_ = tr_data_Y[index1]
                if(class_ not in dict_svs.keys()):
                    dict_svs[class_] = []
                    
                dict_svs[class_].append(tr_data_X[index1])
            
            dual_coefs = clf.dual_coef_
            
            dict_coefs_per_class = {}
            start = 0
            for class_ in dict_svs.keys():
                n_SV_c = len(dict_svs[class_])                                               #number of support vectors for current class
                dict_coefs_per_class[class_] = dual_coefs[:, start:start + n_SV_c]           #Shape: (n_class - 1)*(n_SV_c)
                start = start + n_SV_c
            
            intercepts = clf.intercept_
            all_classes = list(dict_svs.keys())
            series_intercepts = []
            for class1 in range(len(all_classes)):
                for class2 in range(class1 + 1, len(all_classes)):
                    if(class1 == 0):
                        l = [-1, class2]
                    else:
                        l = [class1, class2]
                    series_intercepts.append(l)
            dict_intercepts = {}
            for index1 in range(intercepts.shape[0]):
                intercept = intercepts[index1]
                classes = series_intercepts[index1]
                class1 = classes[0]
                class2 = classes[1]
                if(class1 not in dict_intercepts.keys()):
                    dict_intercepts[class1] = []
                if(class2 not in dict_intercepts.keys()):
                    dict_intercepts[class2] = []
                dict_intercepts[class1].append(intercept)
                dict_intercepts[class2].append(intercept)
           
            #print(dual_coefs[:, :30])
            sum_probs = 0.0
            print("Train data")
            for i in range(tr_samples):
                class_ = calc_data_Y[i]
                dist_sep_plane = dist_separating_hyperplane_general(calc_data_X, params, kernel, calc_data_X[i], class_, dict_coefs_per_class[class_], dict_svs[class_], dict_intercepts[class_])
                l_dist = l_func(dist_sep_plane)
                #phi(x) = x
                h_val = h_general(calc_data_X[i], params, kernel, calc_data_X)
                weights = get_weights(dict_coefs_per_class[class_], dict_svs[class_])
                z_val = Z(weights)
                total_prob = math.exp(-1.0*l_dist) * h_val * z_val
                sum_probs = sum_probs + total_prob
                #print(total_prob)
                train_probs[i][index] = total_prob
                train_class[i] = calc_data_Y[i]
                
            #Actual normalization is being done here. Approximate the normalization term based on the current dataset, as integration over all possible (x,y) pairs is not tractable.  
            train_probs[:, index] = train_probs[:, index]/(sum_probs)
            sum_probs = 0.0
            print("Test data")
            for i in range(te_samples):
                class_ = test_Y[i]
                dist_sep_plane = dist_separating_hyperplane_general(test_X, params, kernel, test_X[i], class_, dict_coefs_per_class[class_], dict_svs[class_], dict_intercepts[class_])
                l_dist = l_func(dist_sep_plane)
                #phi(x) = x
                h_val = h_general(test_X[i], params, kernel, test_X)
                #weights = get_weights(dict_coefs_per_class[class_], dict_svs[class_])
                #z_val = Z(weights)
                z_val = 1.0
                total_prob = math.exp(-1.0*l_dist) * h_val * z_val
                sum_probs = sum_probs + total_prob
                #print(total_prob)
                test_probs[i][index] = total_prob
                test_class[i] = test_Y[i]
            test_probs[:, index] = test_probs[:, index]/(sum_probs)
            
            index = index + 1
            
        else:
            print("Bad Kernel")
            pass
    return train_probs, test_probs, train_class, test_class

def final_computation_WAIC(kernel, tr_data_X, tr_data_Y, calc_data_X, calc_data_Y, test_X, test_Y, classifier, params):
    tr_data_Y[tr_data_Y == 0] = -1
    calc_data_Y[calc_data_Y == 0] = -1
    test_Y[test_Y == 0] = -1
    dict_clfs = get_classifiers_list(kernel, tr_data_X, tr_data_Y, calc_data_X, calc_data_Y, test_X, test_Y)
    dict_clf = get_dict_clf(kernel, classifier, params)
    dict_clfs['clf_list'].append(dict_clf)
    
    no_samples_train = calc_data_X.shape[0]
    no_samples_test = test_X.shape[0]
    classes_ = np.unique(test_Y)
    for class_ in classes_:
        indices = (test_Y == class_).nonzero()
        print(np.array(indices).shape)
        no_queries_per_class = np.array(indices).shape[0]
        #print("Class: ", class_, ", No_samples: ", no_queries_per_class)
    
    #a = input()
    train_probs, test_probs, train_class, test_class = compute_probs(kernel, dict_clfs, no_samples_train, no_samples_test)
    x1, y1 = calculate_waic(train_probs, train_class, True), calculate_waic(test_probs, test_class, True)
    x2, y2 = calculate_waic(train_probs, train_class, False), calculate_waic(test_probs, test_class, False)
    return x1, y1, x2, y2


# =============================================================================
# tr_X, tr_Y, cv_X, cv_Y, te_X, te_Y = load_sentiment_dataset(classification_type = 'multi')
# 
# waic_train, waic_test, waic_train_total, waic_test_total = {}, {}, {}, {}
# 
# params = {'C': 1, 'kernel' : 'linear'}
# clf = svm.SVC(C = params['C'], kernel = params['kernel'], decision_function_shape = 'ovo')
# w_tr, w_te, w_tr_tot, w_te_tot = final_computation_WAIC(params['kernel'], tr_X, tr_Y, tr_X, tr_Y, te_X, te_Y, clf, params)
# waic_train['linear'] = w_tr
# waic_test['linear'] = w_te
# waic_train_total['linear'] = w_tr_tot
# waic_test_total['linear'] = w_te_tot
# 
# print(waic_train, waic_test)
# print(waic_train_total, waic_test_total)
# 
# params = {'C': 1, 'kernel' : 'poly', 'gamma' : 0.1, 'coef0': 0.0, 'degree' : 3}
# clf = svm.SVC(C = params['C'], kernel = params['kernel'], gamma = params['gamma'], coef0 = params['coef0'], degree = params['degree'], decision_function_shape = 'ovo')
# w_tr, w_te, w_tr_tot, w_te_tot = final_computation_WAIC(params['kernel'], tr_X, tr_Y, tr_X, tr_Y, te_X, te_Y, clf, params)
# waic_train['poly'] = w_tr
# waic_test['poly'] = w_te
# waic_train_total['poly'] = w_tr_tot
# waic_test_total['poly'] = w_te_tot
# 
# print(waic_train, waic_test)
# print(waic_train_total, waic_test_total)
# 
# params = {'C': 10, 'kernel' : 'rbf', 'gamma': 1}
# clf = svm.SVC(C = params['C'], kernel = params['kernel'], gamma = params['gamma'], decision_function_shape = 'ovo')
# w_tr, w_te, w_tr_tot, w_te_tot = final_computation_WAIC(params['kernel'], tr_X, tr_Y, tr_X, tr_Y, te_X, te_Y, clf, params)
# waic_train['rbf'] = w_tr
# waic_test['rbf'] = w_te
# waic_train_total['rbf'] = w_tr_tot
# waic_test_total['rbf'] = w_te_tot
# 
# print(waic_train, waic_test)
# print(waic_train_total, waic_test_total)
# =============================================================================
    