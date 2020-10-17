from sklearn import svm
import numpy as np
import math
from fns import load_sentiment_dataset
from scipy.special import beta, betainc

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

def get_classifiers_list(kernel, data_X, data_Y):
    clfs = []
    if(kernel == 'linear'):
        parameters_list = [{'C' : 1}, {'C': 10}, {'C': 100}, {'C': 1000}]
        for params in parameters_list:
            dict_clf = {}
            new_clf = svm.SVC(C = params['C'], kernel = 'linear')
            dict_clf['clf'] = new_clf
            dict_clf['data_X'] = data_X
            dict_clf['data_Y'] = data_Y
            dict_clf['params'] = params
            clfs.append(dict_clf)
        
    elif(kernel == 'rbf'):
        parameters_list = [{'C' : 1, 'gamma': 'scale'}, {'C': 10, 'gamma': 'scale'}, {'C': 100, 'gamma': 'scale'}, {'C': 1000, 'gamma': 'scale'}, 
                           {'C' : 1, 'gamma': 'auto'}, {'C': 10, 'gamma': 'auto'}, {'C': 100, 'gamma': 'auto'}, {'C': 1000, 'gamma': 'auto'}]
        for params in parameters_list:
            dict_clf = {}
            new_data_X = kernel_function(kernel, data_X, params)
            new_clf = svm.SVC(C = params['C'], kernel = 'linear')
            dict_clf['clf'] = new_clf
            dict_clf['data_X'] = new_data_X
            dict_clf['data_Y'] = data_Y
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
            new_data_X = kernel_function(kernel, data_X, params)
            new_clf = svm.SVC(C = params['C'], kernel = 'linear')
            dict_clf['clf'] = new_clf
            dict_clf['data_X'] = new_data_X
            dict_clf['data_Y'] = data_Y
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
            new_data_X = kernel_function(kernel, data_X, params)
            new_clf = svm.SVC(C = params['C'], kernel = 'linear')
            dict_clf['clf'] = new_clf
            dict_clf['data_X'] = new_data_X
            dict_clf['data_Y'] = data_Y
            dict_clf['params'] = params
            clfs.append(dict_clf)
            
    return clfs

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
    if(fn == 'poly'):
        if(params['gamma'] == 'scale'):
            gamma = 1/(x.shape[1] * np.var(x))
        elif(params['gamma'] == 'auto'):
            gamma = 1/(x.shape[1])
        #(gamma *<x, y> + r)^d
        sqrt_gamma = math.sqrt(gamma)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i][j] = sqrt_gamma * x[i][j]
        if(params['degree'] == 2):
            new_no_features = get_new_no_features(x.shape[1], params['degree'])
            new_array = np.empty((x.shape[0], new_no_features))
            
            for i in range(x.shape[0]):
                index = 0
                for j in range(x.shape[1]):
                    new_array[i][index] = (x[i][j])*(x[i][j])
                    index = index + 1
                for j in range(x.shape[1]):
                    for k in range(j+1, x.shape[1]):
                        new_array[i][index] = math.sqrt(2) * (x[i][j] * x[i][k])
                        index = index + 1
                for j in range(x.shape[1]):
                    new_array[i][index] = math.sqrt(2) * x[i][j] * params['coef0']
                    index = index + 1
                new_array[i][index] = params['coef0']
            return new_array
    elif(fn == 'sigmoid'):
        return x
    elif(fn == 'rbf'):
        return x
    else:
        return x           

#h should be a positive integrable function such that h(x1) = h(x2) for ||x1|| = ||x2||
#Examples given in paper: h(x) = exp(-<x, c1Ex>), h(x) = c2[||x||< c3] where c1, c2, c3 are arbitrary strictly positive scalars and E is the identity matrix
def h(x):
    E = np.identity(x.shape[0], dtype = float)
    c1 = 2.0
    outp = np.matmul(E, x)
    outp = c1 * outp
    outp = np.dot(x, outp)
    return math.exp(-1.0 * outp)

def dist_separating_hyperplane(x, weights, y, b):
    if(y == 0):
        y = -1
    return (np.dot(weights, x) + b) * y

def l(d):
    return max(1-d, 0)

#Z has to be used as a normalization constant.
#Based on paper, Z = 1/(integration over all x of the h_val*e^(-l_dist) values.) See paper for clarity.
def Z(weights):
    norm_val = np.linalg.norm(weights)
    return 1/(norm_val + 1e-9)
    
def compute_probs(kernel, data_X, data_Y):
    clf_list = get_classifiers_list(kernel, data_X, data_Y)
    no_sampled_parameters = len(clf_list)

    probs = np.empty((no_sampled_parameters, data_X.shape[0]))
    index = 0
    
    for classifier in clf_list:
        clf = classifier['clf']
        new_data_X = classifier['data_X']
        new_data_Y = classifier['data_Y']
        
        clf.fit(new_data_X, new_data_Y)
        if(kernel == 'linear'):
            #Works for 2-class classification only
            w = clf.coef_[0]
            b = clf.intercept_[0]
            #data transformed to higher dimensions, here =x as it is linear kernel
            sum_probs = 0.0
            for i in range(data_X.shape[0]):
                dist_sep_plane = dist_separating_hyperplane(new_data_X[i], w, new_data_Y[i], b)
                l_dist = l(dist_sep_plane)
                #phi(x) = x
                h_val = h(new_data_X[i])
                z_val = Z(w)
                total_prob = math.exp(-1.0*l_dist) * h_val * z_val
                sum_probs = sum_probs + total_prob
                print(total_prob)
                probs[index][i] = total_prob
            #Actual normalization is being done here. Approximate the normalization term based on the current dataset, as integration over all possible (x,y) pairs is not tractable.  
            probs[index] = probs[index]/(sum_probs)
            index = index + 1
        elif(kernel == 'rbf'):
            pass
        elif(kernel == 'poly'):
            pass
        elif(kernel == 'sigmoid'):
            pass
        else:
            print("Bad Kernel")
            pass
    return probs

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

    
def compute_probs_2nd_method(kernel, data_X, data_Y):
    clf_list = get_classifiers_list(kernel, data_X, data_Y)
    no_sampled_parameters = len(clf_list)

    probs = np.empty((no_sampled_parameters, data_X.shape[0]))
    index = 0
    
    for classifier in clf_list:
        clf = classifier['clf']
        new_data_X = classifier['data_X']
        new_data_Y = classifier['data_Y']
        
        clf.fit(new_data_X, new_data_Y)
        #Works for 2-class classification only
            

tr_X, tr_Y, cv_X, cv_Y, te_X, te_Y = load_sentiment_dataset()

probs = compute_probs('linear', tr_X, tr_Y)
waic = calculate_waic(probs)

print(waic)