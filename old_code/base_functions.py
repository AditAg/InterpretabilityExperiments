#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 23:07:41 2019

@author: adit
"""
import numpy as np
from fns import load_cifar10_dataset, load_mnist_dataset, load_fashionmnist, find_average, get_frequency_components_dataset, load_stanford40_dataset, load_original_sentiment_dataset
from sklearn.model_selection import KFold, train_test_split
import all_globals
from model_classes import ModelA, ModelB
import tensorflow as tf
import cv2

#Problem: A single layer PWLN here, gives a very high accuracy on the MNIST dataset, so effectively, interpretability will still be near to 1 as both A and B are able to
#match the decision boundary for MNIST very accurately.

#Test X and Test Y donot play any role here.
#CUrrently CNN implemented only for CIFAR-10 and might give errors for MNIST or other datasets.

def get_dataset(dataset_name, is_binarized, is_resized, is_grayscale):
    if(dataset_name == "sentiment_analysis"):
        data_X, data_Y, cv_X, cv_Y, test_X, test_Y = load_original_sentiment_dataset()
        return data_X, data_Y, cv_X, cv_Y, test_X, test_Y
    if (dataset_name == "mnist"):
        data_X, data_Y, test_X, test_Y = load_mnist_dataset(is_binarized, is_resized)
    elif(dataset_name == 'cifar-10'):
        data_X, data_Y, test_X, test_Y = load_cifar10_dataset(is_grayscale, is_resized)
        #print("Dataset loaded")
    elif(dataset_name == 'fashion_mnist'):
        data_X, data_Y, test_X, test_Y = load_fashionmnist(is_binarized, is_resized)
    elif(dataset_name == 'stanford40'):
        data_X, data_Y, test_X, test_Y, data_X_A, test_X_A = load_stanford40_dataset()
        return data_X, data_Y, test_X, test_Y, data_X_A, test_X_A
    else:
        print("Not implemented yet")
    return data_X, data_Y, test_X, test_Y
        
def perform_interpretation(mc_dropout, outp_train, outp_cv, outp_test, model_A, model_B):
    dump_bool = False
    if(mc_dropout == False):
        no_repeats = 1
    else:
        no_repeats = 5
    for iters in range(no_repeats):
        predictions_train_B, predictions_cv_B, predictions_test_B = model_B.get_output()
        if(iters == 0):
            dump_bool = True
        else:
            dump_bool = False
        interpretability_train, interpretability_cv, interpretability_test = model_A.calculate_interpretability(predictions_train_B, predictions_cv_B, predictions_test_B, dump_bool)
        #print("The Interpretability of the black box model B on Train Dataset is: ", interpretability_train)
        #print("The Interpretability of the black box model B on CrossValidation Dataset is: ", interpretability_cv)
        #print("The Interpretability of the black box model B on Test Dataset is: ", interpretability_test)
        outp_train.append(interpretability_train)
        outp_cv.append(interpretability_cv)
        outp_test.append(interpretability_test)
        print("Values are:",interpretability_train, interpretability_cv, interpretability_test)
        #print_to_file(interpretability_train, interpretability_cv, interpretability_test)
    #predictions_train_B = np.zeros((5, X_train.shape[0]), dtype = int)
    #predictions_cv_B = np.zeros((5, X_cross_validation.shape[0]), dtype = int)
    #    model_A.fit_model_to_initial_dataset()
    return outp_train, outp_cv, outp_test
 
def print_to_file(train_val, cv_val, test_val):
    file1 = open("Exp1_No_samples/output_mnist_gini_bestsplit_5.txt", 'a')
    #file1.write("\nTrain " + str(train_val))
    #file1.write("\nCV " + str(cv_val))
    file1.write("\nTest " + str(test_val) + "\n")
    file1.close()

def test_kfold_cross_validation(no_samples, sess):
    data_X, data_Y, test_X, test_Y = get_dataset(all_globals.dataset_name, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
    print(data_X.shape, data_Y.shape)
    data_X, data_Y = data_X[:no_samples[0]], data_Y[:no_samples[0]]
    ##print(data_X.shape, data_Y.shape)
    print(np.unique(data_Y))
    final_train, final_cv, final_test = [], [], [] 
    interpret_train = []
    interpret_test = []
    interpret_cv = []
    for itera in range(2):
        print("Iteration" + str(itera))
        kf = KFold(n_splits=all_globals.no_of_folds, shuffle = True, random_state = None)
        i = 1
        outp_train, outp_cv, outp_test = [], [], []
        for train_index, cross_validation_index in kf.split(data_X):
            print("Cross Validation fold " + str(i))
            #print("TRAIN INDEXES:", train_index, "CROSS_VALIDATION INDEXES:", cross_validation_index)
            model_B = ModelB(sess, all_globals.model_B_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_B, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
            model_A = ModelA(sess, all_globals.model_A_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_A, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
            model_B.init_model()
        
            X_train, X_cross_validation = data_X[train_index], data_X[cross_validation_index]
            Y_train, Y_cross_validation = data_Y[train_index], data_Y[cross_validation_index]
            model_B.set_dataset(X_train, Y_train, test_X, test_Y, X_cross_validation, Y_cross_validation)
            model_B.train_model()
            ##print("Model B trained")
            model_A.set_dataset(X_train, Y_train, test_X, test_Y, X_cross_validation, Y_cross_validation)
            outp_train, outp_cv, outp_test = perform_interpretation(all_globals.mc_dropout, outp_train, outp_cv, outp_test, model_A, model_B)
            i = i + 1
        print("Train ", find_average(outp_train))
        print("CV ", find_average(outp_cv))
        print("Test ", find_average(outp_test))
        final_train.extend(outp_train)
        final_cv.extend(outp_cv)
        final_test.extend(outp_test)
        final_train.append(find_average(outp_train))
        final_cv.append(find_average(outp_cv))
        final_test.append(find_average(outp_test))
        
        #print_to_file(find_average(outp_train), find_average(outp_cv), find_average(outp_test))
        interpret_train.append(find_average(outp_train))
        interpret_test.append(find_average(outp_test))
        interpret_cv.append(find_average(outp_cv))

    print("Final Interpretability on CV", find_average(interpret_cv))
    print("Final interpretability on Train", find_average(interpret_train))
    print("Final interpretability on Test", find_average(interpret_test))
    final_train.append(find_average(interpret_train))
    final_cv.append(find_average(interpret_cv))
    final_test.append(find_average(interpret_test))
    print(final_train)
    print(final_cv)
    print(final_test)
    #print_to_file(final_train, final_cv, final_test)
    
def test_cross_validation(no_samples, sess):
    data_X, data_Y, test_X, test_Y = get_dataset(all_globals.dataset_name, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
    print(data_X.shape, data_Y.shape)
    data_X, data_Y = data_X[:no_samples[0]], data_Y[:no_samples[0]]
    ##print(data_X.shape, data_Y.shape)
    print(np.unique(data_Y))
    
    interpret_train = []
    interpret_test = []
    interpret_cv = []
    for itera in range(3):
        model_B = ModelB(sess, all_globals.model_B_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_B, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
        model_A = ModelA(sess, all_globals.model_A_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_A, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
        model_B.init_model()
        X_train, X_cross_validation, Y_train, Y_cross_validation = train_test_split(data_X, data_Y, test_size = 0.2, random_state = 42, stratify = data_Y)
        print("Iteration " + str(itera))
        model_B.set_dataset(X_train, Y_train, test_X, test_Y, X_cross_validation, Y_cross_validation)
        model_B.train_model()
        ##print("Model B trained")
        model_A.set_dataset(X_train, Y_train, test_X, test_Y, X_cross_validation, Y_cross_validation)
        outp_train, outp_cv, outp_test = [], [], []
        outp_train, outp_cv, outp_test = perform_interpretation(all_globals.mc_dropout, outp_train, outp_cv, outp_test, model_A, model_B)
        print("Train", find_average(outp_train))
        print("CV", find_average(outp_cv))
        print("Test", find_average(outp_test))
        interpret_train.append(find_average(outp_train))
        interpret_test.append(find_average(outp_test))
        interpret_cv.append(find_average(outp_cv))
        #tf.compat.v1.reset_default_graph() <-------- THIS DOESN'T WORK, gives error: AssertionError: Do not use tf.reset_default_graph() to clear nested graphs. If you need a cleared graph, exit the nesting and create a new graph.

    print("Final Interpretability on CV", find_average(interpret_cv))
    print("Final interpretability on Train", find_average(interpret_train))
    print("Final interpretability on Test", find_average(interpret_test))

def test_kfold_cross_validation_stanford40(sess):
    data_X, data_Y, test_X, test_Y, data_X_A, test_X_A = get_dataset('stanford40', all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
    ##print(data_X.shape, data_Y.shape)
    print(np.unique(data_Y))
    
    interpret_train = []
    interpret_test = []
    interpret_cv = []
    for itera in range(3):
        kf = KFold(n_splits=all_globals.no_of_folds, shuffle = True, random_state = None)
        outp_train, outp_cv, outp_test = [], [], []
        indices1 = list(kf.split(data_X))
        indices2 = list(kf.split(data_X_A))
        for i in range(len(indices1)):
            model_B = ModelB(sess, all_globals.model_B_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_B, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
            model_A = ModelA(sess, all_globals.model_A_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_A, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
            model_B.init_model()
            print("Cross Validation fold " + str(i+1))
            #print("TRAIN INDEXES:", train_index, "CROSS_VALIDATION INDEXES:", cross_validation_index)
            train_index, cross_validation_index = indices1[i]
            X_train, X_cross_validation = data_X[train_index], data_X[cross_validation_index]
            Y_train, Y_cross_validation = data_Y[train_index], data_Y[cross_validation_index]
            model_B.set_dataset(X_train, Y_train, test_X, test_Y, X_cross_validation, Y_cross_validation)
            model_B.train_model()
            ##print("Model B trained")
            train_index, cross_validation_index = indices1[i]
            X_train_A, X_cross_validation_A = data_X_A[train_index], data_X_A[cross_validation_index]
            Y_train_A, Y_cross_validation_A = data_Y[train_index], data_Y[cross_validation_index]
            model_A.set_dataset(X_train_A, Y_train_A, test_X_A, test_Y, X_cross_validation_A, Y_cross_validation_A)
            outp_train, outp_cv, outp_test = perform_interpretation(all_globals.mc_dropout, outp_train, outp_cv, outp_test, model_A, model_B)
        print("Train", find_average(outp_train))
        print("CV", find_average(outp_cv))
        print("Test", find_average(outp_test))
        #print_to_file(find_average(outp_train), find_average(outp_cv), find_average(outp_test))
        interpret_train.append(find_average(outp_train))
        interpret_test.append(find_average(outp_test))
        interpret_cv.append(find_average(outp_cv))

    print("Final Interpretability on CV", find_average(interpret_cv))
    print("Final interpretability on Train", find_average(interpret_train))
    print("Final interpretability on Test", find_average(interpret_test))
    #print_to_file(find_average(interpret_train), find_average(interpret_cv), find_average(interpret_test))
    
def test_sentiment_analysis(no_samples, sess):
    data_X, data_Y, cv_X, cv_Y, test_X, test_Y = get_dataset(all_globals.dataset_name, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
    no_features = data_X.shape[1]
    print(data_X.shape, data_Y.shape)
    data_X, data_Y = data_X[:no_samples[0]], data_Y[:no_samples[0]]
    ##print(data_X.shape, data_Y.shape)
    print(np.unique(data_Y))
    
    interpret_train = []
    interpret_test = []
    interpret_cv = []
    for itera in range(3):
        model_B = ModelB(sess, all_globals.model_B_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_B, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale, no_features)
        model_A = ModelA(sess, all_globals.model_A_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_A, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale, no_features)
        model_B.init_model()
        #X_train, X_cross_validation, Y_train, Y_cross_validation = train_test_split(data_X, data_Y, test_size = 0.2, random_state = 42, stratify = data_Y)
        print("Iteration " + str(itera))
        model_B.set_dataset(data_X, data_Y, test_X, test_Y, cv_X, cv_Y)
        model_B.train_model()
        ##print("Model B trained")
        model_A.set_dataset(data_X, data_Y, test_X, test_Y, cv_X, cv_Y)
        outp_train, outp_cv, outp_test = [], [], []
        outp_train, outp_cv, outp_test = perform_interpretation(all_globals.mc_dropout, outp_train, outp_cv, outp_test, model_A, model_B)
        print("Train", find_average(outp_train))
        print("CV", find_average(outp_cv))
        print("Test", find_average(outp_test))
        interpret_train.append(find_average(outp_train))
        interpret_test.append(find_average(outp_test))
        interpret_cv.append(find_average(outp_cv))
        #tf.compat.v1.reset_default_graph() <-------- THIS DOESN'T WORK, gives error: AssertionError: Do not use tf.reset_default_graph() to clear nested graphs. If you need a cleared graph, exit the nesting and create a new graph.

    print("Final Interpretability on CV", find_average(interpret_cv))
    print("Final interpretability on Train", find_average(interpret_train))
    print("Final interpretability on Test", find_average(interpret_test))
#Finds the frequency component of the dataset which is being used by the unknown model
def interpretation_diff_frequencies(no_samples, sess):
    frequencies_list_width = [10]
    data_X, data_Y, test_X, test_Y = get_dataset(all_globals.dataset_name, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
    data_X, data_Y = data_X[:no_samples[0]], data_Y[:no_samples[0]]
    
    for width in frequencies_list_width:
        print("Width: " + str(width) + "\n")
        high_interpret_train, high_interpret_test, high_interpret_cv = [], [], []
        low_interpret_train, low_interpret_test, low_interpret_cv = [], [], []
        for itera in range(3):
            kf = KFold(n_splits=all_globals.no_of_folds, shuffle = True)
            i = 1
            high_outp_train, high_outp_cv, high_outp_test = [], [], []
            low_outp_train, low_outp_cv, low_outp_test = [], [], []
            for train_index, cross_validation_index in kf.split(data_X):
                model_B = ModelB(sess, all_globals.model_B_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_B, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
                model_A = ModelA(sess, all_globals.model_A_epochs, all_globals.batch_size, all_globals.dataset_name, all_globals.learning_rate, all_globals.model_name_A, all_globals.is_binarized, all_globals.is_resized, all_globals.is_grayscale)
                model_B.init_model()
                print("Cross Validation fold " + str(i))
                X_train, X_cross_validation = data_X[train_index], data_X[cross_validation_index]
                Y_train, Y_cross_validation = data_Y[train_index], data_Y[cross_validation_index]
                
                high_freq_train, low_freq_train = get_frequency_components_dataset(X_train, width)
                high_freq_test, low_freq_test = get_frequency_components_dataset(test_X, width)
                high_freq_cv, low_freq_cv = get_frequency_components_dataset(X_cross_validation, width)
                
                model_B.set_dataset(X_train, Y_train, test_X, test_Y, X_cross_validation, Y_cross_validation)
                model_B.train_model()
                ##print("Model B trained")
                
                model_A.set_dataset(high_freq_train, Y_train, high_freq_test, test_Y, high_freq_cv, Y_cross_validation)
                high_outp_train, high_outp_cv, high_outp_test = perform_interpretation(all_globals.mc_dropout, high_outp_train, high_outp_cv, high_outp_test)
                
                model_A.set_dataset(low_freq_train, Y_train, low_freq_test, test_Y, low_freq_cv, Y_cross_validation)
                low_outp_train, low_outp_cv, low_outp_test = perform_interpretation(all_globals.mc_dropout, low_outp_train, low_outp_cv, low_outp_test, model_A, model_B)
        
                i = i + 1
            print("High_Freq:\nTrain: " +  str(find_average(high_outp_train)) + "\nCV: " + str(find_average(high_outp_cv)) + "\nTest: " + str(find_average(high_outp_test)))
            print("Low_Freq:\nTrain: " +  str(find_average(low_outp_train)) + "\nCV: " + str(find_average(low_outp_cv)) + "\nTest: " + str(find_average(low_outp_test)))
            high_interpret_train.append(find_average(high_outp_train))
            high_interpret_test.append(find_average(high_outp_test))
            high_interpret_cv.append(find_average(high_outp_cv))
            low_interpret_train.append(find_average(low_outp_train))
            low_interpret_test.append(find_average(low_outp_test))
            low_interpret_cv.append(find_average(low_outp_cv))
    
        print("Final Values\nTrain: " + str(find_average(high_interpret_train)) + "\nCV: " + str(find_average(high_interpret_cv)) + "\nTest: " + str(find_average(high_interpret_test)))
        print("Final Values\nTrain: " + str(find_average(low_interpret_train)) + "\nCV: " + str(find_average(low_interpret_cv)) + "\nTest: " + str(find_average(low_interpret_test)))
        
    
    
