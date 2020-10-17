#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 20:53:00 2020

@author: adit
"""
# FILE CONTAINING ALL THE GLOBAL PARAMETERS 

dataset_name = 'sentiment_analysis'
# Possible Model Names - "svm", "ann", "cnn", "inceptionV3", "naive_bayes", "ensemble"
model_name_A = 'svm'
model_name_B = 'ann'
# No of folds for K-fold crossvaliation
no_of_folds = 5
# Whether to resize the dataset or not. If set to true, image is resized from (28, 28) in case of MNIST and Fashion-MNIST to (10, 10)
is_resized = False
# epochs to be used for training the individual models
model_A_epochs = 50
model_B_epochs = 50
learning_rate = 0.001
# Whether to convert images to grayscale or not.
is_grayscale = False
batch_size = 128
# whether to use monte-carlo dropout for neural networks (only supported in anns).
mc_dropout = False
# precision to be used when calculating the model entropies before and after the process of interpretation.
entropy_precision = 1e-9
# these two parameters are to be used only for the experiment where we calculate interpretability and entropy on either original or counter-factual dataset.
# This is an experimental work on NLP based data.
# this parameter is not considered for the current paper evaluation but can be experimented with for future work.
# Based on following paper: https://arxiv.org/pdf/1909.12434.pdf
# Possible values: "original", "counter_factual"
# If counter_factual the data used for training and that used for calculating the interpretability are different. The data
interpretability_mode = 'original'
entropy_mode = 'original'
# this parameter is used if we want to convert the classification task (10 digits) to a binary-classification task
is_binarized = True
# is binary by default, it can be set to 'multi' for a multi-classification task
# this parameters are used for experimentation with sentiment-analysis dataset.
# classification type = multi -> considers a 5-class sentiment classification task, else a 2-class sentiment classification task
classification_type = 'binary'



