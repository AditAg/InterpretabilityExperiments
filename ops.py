#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:40:27 2019

@author: aagarwal
"""

import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dense, GlobalAveragePooling2D
from keras import regularizers, initializers
from sklearn.tree import DecisionTreeClassifier
from keras.preprocessing.image import ImageDataGenerator
from sklearn.feature_selection import SelectKBest, f_classif
from math import ceil
from sklearn import svm, datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
#from h2o4gpu.tree import DecisionTreeClassifier
from keras import optimizers
from keras.utils import to_categorical
from fns import load_svhn, convert_to_all_classes_array, load_sentiment_dataset
from keras import backend as K
import matplotlib.pyplot as plt
#from inception_from_scratch import inceptionv3 as incp
from keras.applications.inception_v3 import InceptionV3
#from keras.models import Model

def convert_one_hot(labels, num_classes):
    y_vector = np.zeros((labels.shape[0], num_classes), dtype=np.float)
    for index, label in enumerate(labels):
        y_vector[index, int(label)] = 1.0
    return y_vector

def dump_data(data_file, **kwargs):
    #if(!append):
    #    f_handle = open(data_file, 'wb')
    arguments = ''
    for key, val in kwargs.items():
        arguments += str(val['name'])
        arguments += "=kwargs['" + str(key) + "']['val'], "
    arguments = arguments.strip()
    arguments = arguments[:-1]
    print(arguments)
    #args = dict(e.split('=') for e in arguments.split(', '))
    #eval("np.savez_compressed(self.dump_file_handle, " + arguments + ")")
    eval("np.savez_compressed(data_file, " + arguments + ")")
    print("Data dumped")

def get_plot(history):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Plot")
    #fig.legend(['Train', 'Test'], loc='upper left')
    ax1.plot(history.history['acc'])
    ax1.plot(history.history['val_acc'])
    ax1.set_title('Model accuracy')
    ax1.set(ylabel = 'Accuracy', xlabel = 'Epoch')
    ax1.legend(['Train', 'Test'], loc='upper left')
    
    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set(ylabel = 'Loss', xlabel = 'Epoch')
    ax2.legend(['Train', 'Test'], loc='upper left')
    plt.subplots_adjust(wspace = 0.5)
    #plt.savefig(os.path.join("plots", name + ".png"))
    plt.show()
    
    
def fit_model_to_initial_dataset(dataset_name, classifier, model_name, is_resized, is_grayscale, pca = None):
        if(dataset_name == 'mnist' or dataset_name == 'fashion_mnist'):
            digits = datasets.load_digits()
            if(is_resized):
                image_shape = (10, 10)
            else:
                image_shape = (28, 28)
            n_samples = len(digits.images)
            new_images = np.zeros((n_samples,) + image_shape)
            #for i in range(len(digits.images)):
            #    new_images[i] = cv2.resize(digits.images[i], dsize=image_shape, interpolation=cv2.INTER_CUBIC)
            # data_images = digits.images.reshape((n_samples, -1))
            data_images = new_images.reshape((n_samples, -1))
            d_X, t_X, d_Y, t_Y = train_test_split(data_images, digits.target)
        
        elif(dataset_name == 'cifar-10'):
            train_X, train_Y = load_svhn(is_grayscale)
            print(train_X.shape, train_Y.shape)
            n_samples = train_X.shape[0]
            train_X_new = train_X.reshape((n_samples, -1))
            print(train_X_new.shape)
            d_X, t_X, d_Y, t_Y = train_test_split(train_X_new, train_Y)
        
        elif(dataset_name == 'sentiment_analysis'):
            _, _, cv_X, cv_Y, _, _ = load_sentiment_dataset()
            #data_X, data_Y = np.empty((2, cv_X.shape[1])), np.empty((2, ))
            #index = 0
            #for i in range(cv_Y.shape[0]):
            #    if(cv_Y[i] == 0):
            #        data_X[index] = cv_X[i]
            #        data_Y[index] = cv_Y[i]
            #        index = index + 1
            #        break
                    #if(index == 3):
                    #    break
            #for i in range(cv_Y.shape[0]):
            #    if(cv_Y[i] == 1):
            #        data_X[index] = cv_X[i]
            #        data_Y[index] = cv_Y[i]
            #        index = index + 1
            #        break
                    #if(index == 5):
                    #    break
            #data_Y = data_Y.astype('int')
            data_X = np.random.rand(2, cv_X.shape[1])
            data_Y = np.array([0, 1])
            classifier.fit(data_X, data_Y)
            return classifier
        
        if (model_name == 'svm' or model_name == 'knn' or model_name == 'naive_bayes'):
            if(pca!=None):
                data = pca.fit_transform(d_X)
                #print(pca.explained_variance_ratio_)
                print(data.shape)
                #a = input()
                data = data[:2500]
                d_Y = d_Y[:2500]
            else:
                data = d_X[:2500]
                d_Y = d_Y[:2500]
            classifier.fit(data, d_Y)
        #elif(model_name == 'knn'):
        #    classifier.fit(d_X, d_Y)    
        elif (model_name == "dt" or model_name == "lr"):
            d_X = d_X[:2500]
            d_Y = d_Y[:2500]
            classifier.train_model(d_X, d_Y, t_X, t_Y)
        ##print("The size of the Initial dataset on which Model A is trained is: ", d_X.shape)
        #print("Fitted model A to the initial dataset")
        return classifier

def get_svm_parameters_sentiment_analysis():
    tr_X, tr_Y, cv_X, cv_Y, te_X, te_Y = load_sentiment_dataset(mode='original')
    #tuned_parameters = [{'kernel':['rbf'], 'gamma':[1e-3, 1e-4], 'C' : [1, 10, 100, 1000]}, {'kernel' : ['linear'], 'C' : [1, 10, 100, 1000]}]
    tuned_parameters = [{'kernel' : ['linear'], 'C' : [1, 10]}]
    scores = ['precision']
    for score in scores:
        print("# Tuning hyperparameters for %s" % score)
        clf = GridSearchCV(svm.SVC(), tuned_parameters, scoring = '%s_macro' % score)
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
    return parameters

def get_nb_parameters_sentiment_analysis():
    tr_X, tr_Y, cv_X, cv_Y, te_X, te_Y = load_sentiment_dataset(mode='original')
    #tuned_parameters = [{'alpha' : [100, 10, 5, 4, 3, 2, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}]
    tuned_parameters = [{'alpha' : [1, 0.1]}]
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
    return parameters

class Neural_Network(object):
    def __init__(self, epochs, batch_size, learning_rate, input_size, output_classes, hidden_layers,
                 mc_dropout=False, dropout_rate=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_classes = output_classes
        self.hidden_layers = hidden_layers
        self.mc_dropout = mc_dropout
        self.train_dropout_rate = 0.0
        self.dropout_rate = dropout_rate

    def lr_schedule(self, epoch):
        lrate = self.learning_rate
        if epoch > 50:
            lrate = lrate/10
        if epoch > 75:
            lrate = lrate/5
        return lrate
    
    def create_tf_model(self, name):
        # self.model = Sequential()
        no_hidden_layers = len(self.hidden_layers)
        #
        # for i in range(no_hidden_layers):
        #    if(i == 0):
        #        self.model.add(Dense(self.hidden_layers[0], input_dim = self.input_size, activation = 'relu'))
        #    else:
        #        self.model.add(Dense(self.hidden_layers[i], activation = 'relu'))
        #        
        # if(no_hidden_layers == 0):
        #    self.model.add(Dense(self.output_classes, input_dim = self.input_size, activation = 'sigmoid'))
        # else:
        #    self.model.add(Dense(self.output_classes, activation = 'sigmoid'))
        # 
        self.inp = Input(shape=(self.input_size, ))
        for i in range(no_hidden_layers):
            if (i == 0):
                outp = Dense(self.hidden_layers[0], activation='linear',  kernel_initializer = initializers.TruncatedNormal(stddev = 0.1), bias_initializer = initializers.Constant(1))(self.inp)
                #kernel_regularizer = regularizers.l2(0.01)
                #, activity_regularizer = regularizers.l1(0.01)
                #outp = Dense(self.hidden_layers[0], activation='linear')(self.inp)
                #outp = BatchNormalization()(outp)
                outp = Activation('relu')(outp)
            else:
                outp = Dense(self.hidden_layers[i], activation='linear', kernel_initializer = initializers.TruncatedNormal(stddev = 0.1), bias_initializer = initializers.Constant(1))(outp)
                #kernel_regularizer = regularizers.l2(0.01)
                #, activity_regularizer = regularizers.l1(0.01)
                #outp = Dense(self.hidden_layers[i], activation='linear')(outp)
                #outp = BatchNormalization()(outp)
                outp = Activation('relu')(outp)
            outp = Dropout(0.5)(outp, training=self.mc_dropout)

        if (no_hidden_layers == 0):
            outp = Dense(self.output_classes, activation='linear')(self.inp)
            self.predictions = Activation('softmax')(outp)
        else:
            outp = Dense(self.output_classes, activation='linear')(outp)
            self.predictions = Activation('softmax')(outp)
        #self.model = Model(self.inp, outp, name=name + '_keras')
        self.model = Model(self.inp, self.predictions, name=name + '_keras')

        print(self.model.layers[-3].output.shape)
        print(self.model.layers[-2].output.shape)
        self.get_final_layer_model_output = K.function([self.model.layers[0].input], [self.model.layers[-3].output])
        #self.get_preds = K.function([self.model.layers[0].input], [self.predictions])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #loss='kullback_leibler_divergence'
        #self.model.summary()
        #for layer in self.model.layers:
        #    print (layer)

        
    def train_model(self, data_X, data_Y, cv_X, cv_Y):
        #data_X, data_Y = data_X.astype('float32'), data_Y.astype('float32')
        #cv_X, cv_Y = cv_X.astype('float32'), cv_Y.astype('float32')
        data_Y = convert_one_hot(data_Y, self.output_classes)
        cv_Y = convert_one_hot(cv_Y, self.output_classes)
        
        es_callback = EarlyStopping(monitor='val_loss', patience=3)
        print(data_X.shape, data_Y.shape, cv_X.shape, cv_Y.shape)
        #log_dir="logs/CNN/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        #TODO::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::ADD CROSS VALIDATION HERE
        #self.model.fit(data_X, data_Y, epochs = self.no_epochs, batch_size = self.batch_size, callbacks=[tensorboard_callback])
        #self.model.fit(data_X, data_Y, epochs = self.no_epochs, batch_size = self.batch_size, validation_data = (cv_X, cv_Y), shuffle = True) 
        
        #selector = SelectKBest(f_classif, k=1000)
        #selected_cv_features = selector.fit_transform(data_X, data_Y)
        #print(selected_cv_features.shape)
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.2)
        #datagen.fit(data_X)
        
        #log_dir="logs/ANN/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        
        #self.model.fit_generator(datagen.flow(data_X, data_Y, batch_size = self.batch_size), epochs = self.no_epochs, 
        #                        validation_data = (cv_X, cv_Y), steps_per_epoch = ceil(data_X.shape[0]/self.batch_size), workers = 4, callbacks = [tensorboard_callback])
        
        print(self.epochs, self.batch_size)
        history = self.model.fit(data_X, data_Y, epochs=self.epochs, batch_size=self.batch_size, validation_data = (cv_X, cv_Y), shuffle = True, verbose = 1,
                                 callbacks=[LearningRateScheduler(self.lr_schedule)])
        #tensorboard_callback,
        #get_plot(history)
        
        scores = self.model.evaluate(data_X, data_Y)
        print("Accuracy obtained is : %.2f%%" % (scores[1] * 100))

    def get_predictions(self, data_X, with_acc, data_Y):
        #data_X = data_X.reshape(data_X.shape[0], self.input_size)
        if (with_acc):
            #probs, acc, final_layer = self.get_preds([data_X])[0], self.model.evaluate(data_X, data_Y)[1]*100, self.get_final_layer_model_output([data_X])[0]
            probs, acc, final_layer = self.model.predict(data_X), self.model.evaluate(data_X, data_Y)[1]*100, self.get_final_layer_model_output([data_X])[0]
            preds = np.argmax(np.array(probs), axis=-1)
            #with self.test_summary_writer.as_default():
            #    tf.summary.scalar('loss', loss, step=epoch)
            #    tf.summary.scalar('accuracy', acc, step=epoch)
                
            return probs, preds, final_layer, acc
        else:
            probs, final_layer = self.get_preds([data_X])[0], self.get_final_layer_model_output([data_X])[0]
            preds = np.argmax(np.array(probs), axis=-1)
            return probs, preds, final_layer

class DecisionTree(object):
    def __init__(self, criterion, splitter, max_depth, min_samples_split):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        
        self.model = DecisionTreeClassifier(criterion = self.criterion, splitter = self.splitter, max_depth = self.max_depth, min_samples_split= self.min_samples_split)
        
    def train_model(self, X_train, y_train, cv_X, cv_Y):
        #TODO::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::ADD CROSS VALIDATION HERE
        self.model.fit(X_train, y_train)
    
    def predict_model(self, X_test):
        preds_probs = np.array(self.model.predict_proba(X_test))
        self.classes_ = self.model.classes_
        return preds_probs
  
class MiniVGGNet:
    @staticmethod
    def build(input_shape, no_classes):
        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), padding = "same", input_shape = input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = -1))
        model.add(Conv2D(32, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = -1))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = -1))
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = -1))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(512, activation = 'relu', name = 'last_layer'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(no_classes, activation = 'softmax', name = 'final_output'))
        
        return model
        
    
class CNN(object):
    def __init__(self, input_size, epochs, batch_size, output_size, learning_rate, model_name = 'base_model'):
        self.input_size = input_size
        self.no_epochs = epochs
        self.batch_size = batch_size
        self.output_classes = output_size
        self.learning_rate = learning_rate
        self.model_name = model_name
        #self.model_name = "MiniVGGNet"
    
    def initialize_model(self):
        if(self.model_name == "MiniVGGNet"):
            self.model = MiniVGGNet.build(self.input_size, self.output_classes)
        else:
            self.model = Sequential()
            self.model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform', padding = 'same', input_shape = self.input_size))
            self.model.add(BatchNormalization(axis = -1))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D((2, 2)))
            self.model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))
            self.model.add(BatchNormalization(axis = -1))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            self.model.add(MaxPooling2D((2, 2)))
            self.model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same'))
            self.model.add(BatchNormalization(axis = -1))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
            self.model.add(MaxPooling2D((2, 2)))
            
            self.model.add(Flatten())
            self.model.add(Dense(512))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform', name = 'last_layer'))
            self.model.add(Dense(self.output_classes, activation = 'softmax', name = 'final_output'))
# =============================================================================
#              self.model = Sequential()
#              self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.input_size))
#              self.model.add(Activation('relu'))
#              self.model.add(Conv2D(32, (3, 3)))
#              self.model.add(Activation('relu'))
#              self.model.add(MaxPooling2D(pool_size=(2, 2)))
#              self.model.add(Dropout(0.25))
#              
#              self.model.add(Conv2D(64, (3, 3), padding='same'))
#              self.model.add(Activation('relu'))
#              self.model.add(Conv2D(64, (3, 3)))
#              self.model.add(Activation('relu'))
#              self.model.add(MaxPooling2D(pool_size=(2, 2)))
#              self.model.add(Dropout(0.25))  
#              self.model.add(Conv2D(128, (3, 3), padding='same'))
#              self.model.add(Activation('relu'))
#              self.model.add(Conv2D(128, (3, 3)))
#              self.model.add(Activation('relu'))
#              self.model.add(MaxPooling2D(pool_size=(2, 2)))
#              self.model.add(Dropout(0.25))
# 
#              self.model.add(Flatten())
#              self.model.add(Dense(512, name = 'last_layer'))
#              self.model.add(Activation('relu'))
#              self.model.add(Dropout(0.5))
#              self.model.add(Dense(self.output_classes, activation = 'softmax', name = 'final_output'))
# =============================================================================
        self.output_layer = self.model.get_layer('final_output').output
        self.last_layer = self.model.get_layer('last_layer').output
        self.probs_layer_model = Model(inputs = self.model.input, outputs = self.output_layer)
        self.last_layer_model = Model(inputs = self.model.input, outputs = self.last_layer)
        #self.optimizer = optimizers.Adam(lr = 0.001)
        #self.optimizer = optimizers.SGD(lr = self.learning_rate, momentum = 0.9)
        self.optimizer = optimizers.RMSprop(lr = 0.0001, decay = 1e-6)
        self.model.compile(optimizer = self.optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
    def train_model(self, data_X, data_Y, cv_X, cv_Y):
        data_X, data_Y = data_X.astype('float32'), data_Y.astype('float32')
        cv_X, cv_Y = cv_X.astype('float32'), cv_Y.astype('float32')
        data_Y = convert_one_hot(data_Y, self.output_classes)
        cv_Y = convert_one_hot(cv_Y, self.output_classes)
        
        print(data_X.shape, data_Y.shape, cv_X.shape, cv_Y.shape)
        #log_dir="logs/CNN/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        #TODO::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::ADD CROSS VALIDATION HERE
        #self.model.fit(data_X, data_Y, epochs = self.no_epochs, batch_size = self.batch_size, callbacks=[tensorboard_callback])
        #self.model.fit(data_X, data_Y, epochs = self.no_epochs, batch_size = self.batch_size, validation_data = (cv_X, cv_Y), shuffle = True) 
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
        datagen.fit(data_X)
        self.model.fit_generator(datagen.flow(data_X, data_Y, batch_size = self.batch_size), epochs = self.no_epochs, verbose=0,
                                 validation_data = (cv_X, cv_Y), steps_per_epoch = ceil(data_X.shape[0]/self.batch_size), workers = 4)
        
    def get_predictions(self, data_X, with_acc, data_Y):
        probs = self.probs_layer_model.predict(data_X)
        final_layer = self.last_layer_model.predict(data_X)
        preds = np.argmax(np.array(probs), axis=-1)
        if (with_acc):
            data_Y = to_categorical(data_Y)
            _, acc = self.model.evaluate(data_X, data_Y, verbose = 0)
            return probs, preds, final_layer, acc
        else:
            return probs, preds, final_layer    

class LogisticRegr(object):
    def __init__(self, solver):
        self.model = LogisticRegression(n_jobs = 10, multi_class='multinomial', solver = solver)
    
    def train_model(self, data_X, data_Y, cv_X, cv_Y):
        self.model.fit(data_X, data_Y)
        
    def predict_model(self, data_X):
        preds_probs = np.array(self.model.predict_proba(data_X))
        self.classes_ = self.model.classes_
        return preds_probs
        
class ensemble_model(object):
    def __init__(self, input_size, epochs, batch_size, output_size, learning_rate, ensembling_technique):
        self.input_size = input_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.ensembling_technique = ensembling_technique

    
    def get_model(self, model_name, index, dataset_name, is_resized, is_grayscale):
        self.dataset_name = dataset_name
        if(model_name[0] == 'ann'):
            model = Neural_Network(self.epochs, self.batch_size, self.learning_rate, self.input_size,
                                     self.output_size, model_name[1], mc_dropout =  False, dropout_rate = None)
            model.create_tf_model("ensemble_model" + str(index))
        
        elif(model_name[0] == 'dt'):
            model = DecisionTree(model_name[1], model_name[2], model_name[3], model_name[4])
            model = fit_model_to_initial_dataset(dataset_name, model, "dt", is_resized, is_grayscale)
        
        elif(model_name[0] == 'cnn'):
            model = CNN(self.input_size, self.epochs, self.batch_size, self.output_size, self.learning_rate)
            model.initialize_model()
            
        elif(model_name[0] == 'svm'):
            #self.pca = PCA(n_components = 2000, whiten = True)
            self.pca = None
            #, whiten = True
            param_list = model_name[1]
            model = svm.SVC(C = param_list['C'], kernel = param_list['kernel'], gamma = param_list['gamma'], probability = param_list['probability'], random_state = param_list['random_state'])
            model = fit_model_to_initial_dataset(dataset_name, model, model_name[0], is_resized, is_grayscale, pca = self.pca)
        
        elif(model_name[0] == 'knn'):
            model = KNeighborsClassifier(30)
            self.pca2 = PCA(n_components = 30, whiten = True)
            model = fit_model_to_initial_dataset(dataset_name, model, model_name[0], is_resized, is_grayscale, pca = self.pca2)
        
        elif(model_name[0] == "lr"):
            model = LogisticRegr("lbfgs")
            model = fit_model_to_initial_dataset(dataset_name, model, model_name[0], is_resized, is_grayscale)
        self.models.append((model_name[0], model))
      
    def fit_model(self, is_resized, is_grayscale):
        for index in range(len(self.models)):
            model = list(self.models[index])
            if(model[0] == "svm"):
                model[1] = fit_model_to_initial_dataset(self.dataset_name, model[1], model[0], is_resized, is_grayscale, pca = self.pca)
            elif(model[0] == 'knn'):
                model[1] = fit_model_to_initial_dataset(self.dataset_name, model[1], model[0], is_resized, is_grayscale, pca = self.pca2)
            elif(model[0] == "dt" or model[0] == "lr"):
                model[1] = fit_model_to_initial_dataset(self.dataset_name, model[1], model[0], is_resized, is_grayscale)
            self.models[index] = tuple(model)
                
            
    def initialize_ensemble(self, models_names_list, dataset_name, is_resized, is_grayscale):
        self.models = []
        for index, i in enumerate(models_names_list):
            self.get_model(i, index, dataset_name, is_resized, is_grayscale)
        
        
    def train_ensemble(self, data_X, data_Y, cv_X, cv_Y):
        for index in range(len(self.models)):
            if(self.models[index][0] == 'svm'):
                #print(data_Y.shape)
                #transformed_data = self.pca.fit_transform(data_X)
                transformed_data = data_X
                #TODO:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::ADD CROSS VALIDATION HERE
                #self.models[index][1].fit(data_X, data_Y)
                self.models[index][1].fit(transformed_data, data_Y)
                #print("Fitted the SVM (part of ensemble) to the predictions of model B")
            elif(self.models[index][0] == 'knn'):
                transformed_data = self.pca2.fit_transform(data_X)
                #transformed_data = data_X
                self.models[index][1].fit(transformed_data, data_Y)
            else:
                self.models[index][1].train_model(data_X, data_Y, cv_X, cv_Y)
                #print("Fitted the model (part of the ensemble) to the predictions of model B")
    
    def apply_ensembling(self, predictions):
        #Averaging
        length = len(predictions)
        pred = predictions[0]
        for index in range(1, len(predictions)):
            pred = np.add(pred, predictions[index])
        return np.true_divide(pred, length)
    
    
    def predict_ensemble(self, data_X, with_acc, data_Y):
        #print(data_Y.shape)
        predictions = []
        for index in range(len(self.models)):
            if(self.models[index][0] == "svm" or self.models[index][0] == "dt" or self.models[index][0] == "knn" or self.models[index][0] == "lr"):
                if(self.models[index][0] == "svm" or self.models[index][0] == "knn"):
                    if(self.models[index][0] == "svm"):
                        #new_data_X = self.pca.fit_transform(data_X)
                        new_data_X = data_X
                        #print(pca.explained_variance_ratio_)
                        print(new_data_X.shape)
                        #a = input()
                        probs2 = np.array(self.models[index][1].predict_proba(new_data_X))
                    else:
                        new_data_X = self.pca2.fit_transform(data_X)
                        #new_data_X = data_X
                        print(new_data_X.shape)
                        probs2 = np.array(self.models[index][1].predict_proba(new_data_X))
                else:
                    probs2 = np.array(self.models[index][1].predict_model(data_X))
                probs = convert_to_all_classes_array(probs2, self.models[index][1].classes_, self.output_size)
                print("Accuracy of " + str(self.models[index][0]) +" classifier:",accuracy_score(np.argmax(data_Y, axis = -1), np.argmax(probs, axis = -1)))
                if(probs2.shape[-1] == self.output_size):
                    assert(probs2.all() == probs.all())

            elif(self.models[index][0] == "ann" or self.models[index][0] == "cnn"):
                probs, _, _, acc = self.models[index][1].get_predictions(data_X, with_acc, data_Y)
                print("Accuracy of "+ str(self.models[index][0]) + " : " + str(acc))
                probs = np.array(probs)
                if (probs.shape[0] == 1):
                    probs = np.squeeze(probs, axis=0)
            
            predictions.append(probs)
        vals = self.apply_ensembling(predictions)
        print("Accuracy of ensemble on the" + self.dataset_name + " dataset is: " + str(accuracy_score(np.argmax(data_Y, axis = -1), np.argmax(vals, axis = -1))))
        return vals

# =============================================================================
# class InceptionV3(object):
#     def __init__(self, width, height, frame_count, lr, output, model_name batch_size):
#         self.width = width
#         self.height = height
#         self.frame_count = frame_count
#         self.lr = lr
#         self.output = output
#         self.model_name = nodel_name
#         self.batch_size = batch_size
#     
#     def initialize_model(self):
#         self.model = incp(self.width, self.height, self.frame_count, self.lr, self.output, self.model_name)
#         
#     def train_model(self, data_X, data_Y, cv_X, cv_Y):
#         self.
#     
#     def get_predictions(self, data_X, with_acc, data_Y):
#         
# =============================================================================
class Inceptionv3(object):
    def __init__(self, output_classes, batch_size, no_epochs, input_image_size):
        self.input_image_size = input_image_size
        self.batch_size = batch_size
        self.no_epochs = no_epochs
        self.output_classes = output_classes
    
    def initialize_model(self):
        self.base_model = InceptionV3(weights = 'imagenet', include_top = False, input_tensor = Input(shape = self.input_image_size))
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation = 'relu', name = 'last_layer')(x)
        self.predictions = Dense(self.output_classes, activation = 'softmax', name = 'final_output')(x)
        
        self.output_layer = self.predictions
        self.last_layer = x
        
        self.probs_layer_model = Model(inputs = self.base_model.input, outputs = self.predictions)
        self.last_layer_model = Model(inputs = self.base_model.input, outputs = self.last_layer)
        
        for layer in self.base_model.layers:
            layer.trainable = False
        self.optimizer = optimizers.RMSprop(lr = 0.0001, decay = 1e-6)
        self.probs_layer_model.compile(optimizer = self.optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
        self.probs_layer_model.summary()
        
        
    def train_model(self, data_X, data_Y, cv_X, cv_Y):
        #data_X, data_Y = data_X.astype('float32'), data_Y.astype('float32')
        #cv_X, cv_Y = cv_X.astype('float32'), cv_Y.astype('float32')
        data_Y = convert_one_hot(data_Y, self.output_classes)
        cv_Y = convert_one_hot(cv_Y, self.output_classes)
        print(data_X.shape, cv_X.shape, data_Y.shape, cv_Y.shape)
        for i in range(cv_X.shape[0]):
            print(data_Y[i]);
            cv2.imshow('image', data_X[i]); 
            cv2.imshow('cropped_image', cv_X[i]); 
            cv2.waitKey(0); 
            cv2.destroyAllWindows()
            a = input()   
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(
            featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
            samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06,
            rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
            shear_range=0., zoom_range=0., channel_shift_range=0., fill_mode='nearest',
            cval=0., horizontal_flip=True, vertical_flip=False, rescale=None,
            preprocessing_function=None, data_format=None, validation_split=0.0)
        datagen.fit(data_X)
        self.probs_layer_model.fit_generator(datagen.flow(data_X, data_Y, batch_size = self.batch_size), epochs = 5, 
                                 validation_data = (cv_X, cv_Y), steps_per_epoch = ceil(data_X.shape[0]/self.batch_size), workers = 4)
        
        for i, layer in enumerate(self.base_model.layers):
            print(i, layer.name)
        
        for layer in self.probs_layer_model.layers[:249]:
            layer.trainable = False
        for layer in self.probs_layer_model.layers[249:]:
            layer.trainable = True
        
        self.new_optimizer = optimizers.SGD(lr = 0.0001, momentum = 0.9)
        self.probs_layer_model.compile(optimizer = self.new_optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        self.probs_layer_model.fit_generator(datagen.flow(data_X, data_Y, batch_size = self.batch_size), epochs = self.no_epochs, 
                                 validation_data = (cv_X, cv_Y), steps_per_epoch = ceil(data_X.shape[0]/self.batch_size), workers = 4)
        
    
    def get_output(self, data_X, with_acc, data_Y):
        probs = self.probs_layer_model.predict(data_X)
        final_layer = self.last_layer_model.predict(data_X)
        preds = np.argmax(np.array(probs), axis = -1)
        if(with_acc):
            data_Y = to_categorical(data_Y)
            _, acc = self.probs_layer_model.evaluate(data_X, data_Y, verbose = 0)
            return probs, preds, final_layer, acc
        else:
            return probs, preds, final_layer
