import numpy as np
import math
import os

from h2o4gpu import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
from ops import convert_one_hot, Neural_Network, DecisionTree, CNN, ensemble_model, fit_model_to_initial_dataset, Inceptionv3
from fns import convert_to_all_classes_array

#from thundersvm import SVC
#from sklearn.linear_model import SGDClassifier
#import h2o4gpu as sklearn
#from sklearn import svm

#KNOWN MODEL
class ModelA(object):
    def __init__(self, sess, epochs, batch_size, dataset_name, learning_rate, model_name, is_binarized, is_resized, is_grayscale, feature_size = None):
        self.sess = sess
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.no_epochs = epochs
        self.batch_size = batch_size
        self.is_resized = is_resized
        self.is_grayscale = is_grayscale
        # self.classifier = SGDClassifier(loss = "hinge", penalty = "l2", max_iter = 5)
        self.is_binarized = is_binarized
        self.no_features = feature_size

        if (dataset_name == 'mnist'):
            if(self.is_resized):
                self.input_image_size = (10, 10, 1)
            else:
                self.input_image_size = (28, 28, 1)
            if (is_binarized):
                self.output_classes = 2
            else:
                self.output_classes = 10
            # self.data_X, self.data_Y, self.test_X, self.test_Y = load_mnist_dataset(self.is_binarized)
        elif (dataset_name == 'cifar-10'):
            if(self.is_grayscale):
                self.input_image_size = (28, 28, 1) if (self.is_resized) else (32, 32, 1)
            else:
                self.input_image_size = (32, 32, 3)
            self.output_classes = 10
            
        elif (dataset_name == 'fashion_mnist'):
            self.input_image_size = (10, 10, 1) if (self.is_resized) else (28, 28, 1)
            self.output_classes = 2 if (self.is_binarized) else 10
            
        elif(dataset_name == 'stanford40'):
            self.input_image_size = (200, 200, 3)
            self.output_classes = 40
        elif(dataset_name == 'sentiment_analysis'):
            self.input_image_size = (self.no_features, )
            self.output_classes = 2
        else:
            print("Not Implemented yet")

        if (self.model_name == "svm"):
            self.classifier = svm.SVC(C=1, kernel='rbf', gamma='auto', probability=True, random_state=0)
            #self.classifier = SVC(kernel = 'rbf', gamma = 'auto', C = 1, probability = True, verbose = True, random_state = 0)
            self.classifier = fit_model_to_initial_dataset(self.dataset_name, self.classifier, "svm", self.is_resized, self.is_grayscale)
            
        elif (self.model_name == "ann"):
            input_size = np.prod(self.input_image_size)
            self.hidden_layers = [256]
            self.NN = Neural_Network(self.sess, self.no_epochs, self.batch_size, self.learning_rate, input_size,
                                     self.output_classes, self.hidden_layers)
            self.NN.create_tf_model("ModelA")
            #self.fit_model_to_initial_dataset()
            
        elif(self.model_name == "dt"):
            self.classifier = DecisionTree('gini', 'best', None, 5)
            print("Gini_Best_5")
            self.classifier = fit_model_to_initial_dataset(self.dataset_name, self.classifier, "dt", self.is_resized, self.is_grayscale)
            
        elif(self.model_name == "ensemble"):
            input_size = np.prod(self.input_image_size)
            self.classifier = ensemble_model(input_size, self.no_epochs, self.batch_size, self.output_classes, self.learning_rate, self.sess, "")
            ensemble_list = [("svm", ), ("lr", ), ("dt", "gini", "best", None, 10)]
            self.classifier.initialize_ensemble(ensemble_list, self.dataset_name, self.is_resized, self.is_grayscale)
            print(ensemble_list)
            #("svm", ), ("ann", [512, 128]), ("knn", ),
            # 
            #self.classifier.initialize_ensemble([("svm", ), ("dt", "gini", "best", None, 10)], self.dataset_name, self.is_resized, self.is_grayscale)
            #print("Ensemble being used: ann of 64 neurons\n")
        elif(self.model_name == "cnn"):
            self.CNN_classifier = CNN(self.input_image_size, self.no_epochs, self.batch_size, self.output_classes, self.learning_rate)
            self.CNN_classifier.initialize_model()
        elif(self.model_name == "inceptionv3"):
            self.inception_classifier = Inceptionv3(self.output_classes, self.batch_size, self.no_epochs, self.input_image_size)
            self.inception_classifier.initialize_model()
        
    def set_dataset(self, train_X, train_Y, test_X, test_Y, cv_X, cv_Y):
        self.data_X, self.data_Y, self.test_X, self.test_Y, self.cross_validation_X, self.cross_validation_Y = train_X, train_Y, test_X, test_Y, cv_X, cv_Y

    def calculate_interpretability(self, predictions_model_B_train, predictions_model_B_cross_validation, predictions_model_B_test, dump_bool):
        if(self.model_name =='ann' or self.model_name == 'svm' or self.model_name == 'dt' or self.model_name == "ensemble"):
            self.data_X = self.data_X.reshape(self.data_X.shape[0], -1)
            self.cross_validation_X = self.cross_validation_X.reshape(self.cross_validation_X.shape[0], -1)
            self.test_X = self.test_X.reshape(self.test_X.shape[0], -1)
        print(self.data_X.shape, self.cross_validation_X.shape)
        if(dump_bool):
            #self.dump_file_handle = open('dumped_data.npz', 'ab')
            input_data_dict = {}
            preds_B_dict = {}
            if(self.model_name == "svm"):
                #input_data_dict['val'] = self.data_X[:10000]                     CHANGE
                #preds_B_dict['val'] = predictions_model_B_train[:10000]          CHANGE
                input_data_dict['val'] = self.data_X
                preds_B_dict['val'] = predictions_model_B_train
            else:
                input_data_dict['val'] = self.data_X
                preds_B_dict['val'] = predictions_model_B_train
            input_data_dict['name'] = 'train_data_X'
            preds_B_dict['name'] = 'model_B_predictions'
            self.dump_data('model_B.npz', input_data_dict = input_data_dict, preds_B_dict = preds_B_dict)
           
        ##print("The size of the final dataset used for interpretation is: ", self.data_X.shape)
        initial_entropy_train = self.calculate_entropy(predictions_model_B_train, dump_bool, 'initial', 0)
        initial_entropy_cv = self.calculate_entropy(predictions_model_B_cross_validation, False, 'initial', 1)
        initial_entropy_test = self.calculate_entropy(predictions_model_B_test, False, 'initial', 2)
        #initial_entropy_train, initial_entropy_cv, initial_entropy_test = None, None, None
        print(initial_entropy_train, initial_entropy_cv, initial_entropy_test)
        self.initialize_model(predictions_model_B_train, predictions_model_B_cross_validation)
        final_entropy_train = self.calculate_entropy(predictions_model_B_train, dump_bool, 'final', 0)
        final_entropy_cv = self.calculate_entropy(predictions_model_B_cross_validation, False, 'final', 1)
        final_entropy_test = self.calculate_entropy(predictions_model_B_test, False, 'final', 2)
        print("The initial entropy on Train Dataset is :", initial_entropy_train)
        print("The initial entropy on Cross Validation Dataset is :", initial_entropy_cv)
        print("The final entropy on Train Dataset is : ", final_entropy_train)
        print("The final entropy on Cross Validation Dataset is : ", final_entropy_cv)
        if(initial_entropy_train == 0.0):
            interpret_train = 1.0
        else:
            interpret_train = (initial_entropy_train - final_entropy_train) / initial_entropy_train
        if(initial_entropy_cv == 0.0):
            interpret_cv = 1.0
        else:
            interpret_cv = (initial_entropy_cv - final_entropy_cv) / initial_entropy_cv
        if(initial_entropy_test == 0.0):
            interpret_test = 1.0
        else:
            interpret_test = (initial_entropy_test - final_entropy_test)/initial_entropy_test
        #self.dump_file_handle.close()
        print(interpret_train, interpret_cv, interpret_test)
        #re-initialize the classifier to original state for SVM and Decision Tree.
        #This is necessary to ensure proper cross-validation.
        if(self.model_name == 'svm' or self.model_name == 'dt'):
            self.classifier = fit_model_to_initial_dataset(self.dataset_name, self.classifier, self.model_name, self.is_resized, self.is_grayscale)
        elif(self.model_name == "ensemble"):
            self.classifier.fit_model(self.is_resized, self.is_grayscale)
        return interpret_train, interpret_cv, interpret_test

    def initialize_model(self, predictions_model_B_train, predictions_model_B_cv):
        if (self.model_name == 'svm'):
            # self.classifier = SGDClassifier(loss = "hinge")
            print(self.data_X.shape, predictions_model_B_train.shape)
            #self.classifier.fit(self.data_X[:10000], predictions_model_B[:10000])                      CHANGE
            #TODO::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::ADD CROSS VALIDATION HERE
            self.classifier.fit(self.data_X, predictions_model_B_train)
            print("Fitted the SVM to the Predictions of Model B")

        elif (self.model_name == 'ann'):
            self.data_X = self.data_X.reshape(self.data_X.shape[0], -1)
            print(self.data_X.shape, predictions_model_B_train.shape, self.cross_validation_X.shape)
            self.NN.train_model(self.data_X, predictions_model_B_train, self.cross_validation_X, predictions_model_B_cv)
            print("Trained the Neural Network Model A on Predictions of Model B")
        
        elif(self.model_name == 'dt'):
            self.data_X = self.data_X.reshape(self.data_X.shape[0], -1)
            self.cross_validation_X = self.cross_validation_X.reshape(self.cross_validation_X.shape[0], -1)
            #preds2 = np.argmax(self.classifier.predict_model(self.data_X), axis = -1)
            #print("Final Accuracy of Model A on MNIST Train Dataset before interpretation is :" + str(metrics.accuracy_score(self.data_Y, preds2)))
            self.classifier.train_model(self.data_X, predictions_model_B_train, self.cross_validation_X, predictions_model_B_cv)
            #print("Fitted the Decision Tree to the predictions of model B")
            #preds = np.argmax(self.classifier.predict_model(self.cross_validation_X), axis = -1)
            #print("Final Accuracy of Model A on MNIST CV Dataset is :" + str(metrics.accuracy_score(self.cross_validation_Y, preds)))
            #preds2 = np.argmax(self.classifier.predict_model(self.data_X), axis = -1)
            #print("Final Accuracy of Model A on MNIST CV Dataset is :" + str(metrics.accuracy_score(self.data_Y, preds2)))
        
        elif(self.model_name == "ensemble"): #cannot have CNN here, as the data_X is being flattened out here, for use by other models
            self.data_X = self.data_X.reshape(self.data_X.shape[0], -1)
            self.cross_validation_X = self.cross_validation_X.reshape(self.cross_validation_X.shape[0], -1)
            self.classifier.train_ensemble(self.data_X, predictions_model_B_train, self.cross_validation_X, predictions_model_B_cv)
        elif(self.model_name == "cnn"):
            self.CNN_classifier.train_model(self.data_X, predictions_model_B_train, self.cross_validation_X, predictions_model_B_cv)
        elif(self.model_name == "inceptionv3"):
            self.inception_classifier.train_model(self.data_X, predictions_model_B_train, self.cross_validation_X, predictions_model_B_cv)
        else:
            print("Not implemented yet")

    def calculate_entropy(self, preds, dump_bool, name, split):
        # prob = np.array(self.classifier.decision_function(self.data_X))
        # prob_B_indexes = np.argmax(predictions_model_B, axis = -1)
        if(split == 0):
            data = self.data_X
            output = self.data_Y
            #print("Train split")
        elif(split == 1):
            data = self.cross_validation_X
            output = self.cross_validation_Y
            #print("Cross Validation split ")
        elif(split == 2):
            data = self.test_X
            output = self.test_Y
            #print("Test split")
        else:
            #print("Invalid Split Value")
            return None
        if (self.model_name == 'svm'):
            #probs_train = np.array(self.classifier.predict_proba(self.data_X[:10000]))                CHANGE
            probs2 = np.array(self.classifier.predict_proba(data))
            probs = convert_to_all_classes_array(probs2, self.classifier.classes_, self.output_classes)
            if(probs2.shape[-1] == self.output_classes):
                assert(probs2.all() == probs.all())
            print("Accuracy of Model A on the current split of the dataset is : ", accuracy_score(output, np.argmax(probs2, axis = -1)))
            
        elif (self.model_name == 'ann'):
            probs, _, _, acc = self.NN.get_predictions(data, True, convert_one_hot(output, self.output_classes))  # These are 1X50000 arrays
            print(probs.shape)
            print("Accuracy of Model A on the" + self.dataset_name + "Training Dataset is: " + str(acc))
            #print("Accuracy of Model A on the MNIST CrossValidation Dataset is: " + str(acc2))
            probs = np.array(probs)
            if (probs.shape[0] == 1):
                probs = np.squeeze(probs, axis=0)
        
        elif(self.model_name == 'cnn' or self.model_name == 'inceptionv3'):
            if(self.model_name == 'inceptionv3'):
                probs, _, _, acc = self.inception_classifier.get_output(data, True, output)
            else:
                probs, _, _, acc = self.CNN_classifier.get_predictions(data, True, output)
            print("Accuracy of Model A on the" + self.dataset_name + "Training Dataset is: " + str(acc))
            probs = np.array(probs)
            if (probs.shape[0] == 1):
                probs = np.squeeze(probs, axis=0)
                
        elif(self.model_name == 'dt'):
            probs2 = self.classifier.predict_model(data)
            #print(probs2[0])
            probs = convert_to_all_classes_array(probs2, self.classifier.classes_, self.output_classes)
            if(probs2.shape[-1] == self.output_classes):
                assert(probs2.all() == probs.all())
            #print("Accuracy of Model A on the current split of the dataset is : ", accuracy_score(output, np.argmax(probs2, axis = -1)))
        elif(self.model_name == "ensemble"):
            probs = np.array(self.classifier.predict_ensemble(data, True, convert_one_hot(output, self.output_classes)))
        
        # actual_probs = np.exp(probs2)/(np.sum(np.exp(probs2), axis = 1))
        prob_A_indexes = np.argmax(probs, axis=-1)
        ##print("Classes predicted by the model A: ", np.unique(prob_A_indexes))
        if(dump_bool):
            name = name + '_model_A_predictions'
            dict1 = {}
            dict1['val'] = prob_A_indexes
            dict1['name'] = name
            self.dump_data(name + '.npz', preds_A_dict = dict1)
            
        ##print(probs_train.shape, preds_train.shape, prob_A_indexes.shape)
        total_diff = 0.0
        list_diff = []
        for i in range(probs.shape[0]):
            if (prob_A_indexes[i] != preds[i]):
                list_diff.append([i, prob_A_indexes[i], preds[i]])
            
            #print(probs[i][prob_A_indexes[i]])
            val = (abs(probs[i][prob_A_indexes[i]] - probs[i][preds[i]]))
            if(val <= 0):
                total_diff += 0.0
            else:
                total_diff += -1.0 * (math.log2(val))

        total_diff = (total_diff) / (probs.shape[0])

        if (len(list_diff) == 0):
            print("For Model A " + str(
                self.model_name) + " and Model B as ANN, the final predictions on this split of the dataset are same")
        else:
            None
            #print("The no of different values are: " + str(len(list_diff)))
            # + " and list is: "
            #print(list_diff)
            
        return total_diff

    def dump_data(self, data_file, **kwargs):
        #if(!append):
        #    f_handle = open(data_file, 'wb')
        arguments = ''
        for key, val in kwargs.items():
            arguments += str(val['name'])
            arguments += "=kwargs['" + str(key) + "']['val'], "
        arguments = arguments.strip()
        arguments = arguments[:-1]
        ##print(arguments)
        #args = dict(e.split('=') for e in arguments.split(', '))
        #eval("np.savez_compressed(self.dump_file_handle, " + arguments + ")")
        eval("np.savez_compressed(data_file, " + arguments + ")")
        ##print("Data dumped")

#unknown model
class ModelB(object):
    def __init__(self, sess, epochs, batch_size, dataset_name, learning_rate, model_name, is_binarized, is_resized, is_grayscale, feature_size = None):
        self.sess = sess
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.no_epochs = epochs
        self.batch_size = batch_size
        self.is_binarized = is_binarized
        self.is_resized = is_resized
        self.is_grayscale = is_grayscale
        self.no_features = feature_size

        if (dataset_name == 'mnist'):
            if(self.is_resized):
                self.input_image_size = (10, 10, 1)
            else:
                self.input_image_size = (28, 28, 1)
            if (is_binarized):
                self.output_classes = 2
            else:
                self.output_classes = 10
            # self.data_X, self.data_Y, self.test_X, self.test_Y = load_mnist(is_binarized)
            # self.cross_validation_X, self.cross_validation_Y = self.data_X, self.data_Y
            # self.no_batches = len(self.data_X)//(self.batch_size)
        elif (dataset_name == 'cifar-10'):
            if(self.is_grayscale):
                self.input_image_size = (28, 28, 1) if (self.is_resized) else (32, 32, 1)
            else:
                self.input_image_size = (32, 32, 3)
            self.output_classes = 10
            
        elif (dataset_name == 'fashion_mnist'):
            self.input_image_size = (10, 10, 1) if (self.is_resized) else (28, 28, 1)
            self.output_classes = 2 if (self.is_binarized) else 10
         
        elif(dataset_name == 'stanford40'):
            self.input_image_size = (200, 200, 3)
            self.output_classes = 40
        elif(dataset_name == 'sentiment_analysis'):
            self.input_image_size = (self.no_features, )
            self.output_classes = 2
        else:
            print("Not Implemented yet")

        if (self.model_name == 'ann'):
            input_size = np.prod(self.input_image_size)
            self.hidden_layers = [512, 256, 128, 64]
            self.NN = Neural_Network(self.sess, self.no_epochs, self.batch_size, self.learning_rate, input_size,
                                     self.output_classes, self.hidden_layers, mc_dropout =  False, dropout_rate = None)

        elif (self.model_name == 'svm'):
            self.classifier = svm.SVC(C=1, kernel='rbf', gamma='auto', probability=True, random_state=0)
            #self.classifier = SVC(kernel = 'rbf', gamma = 'auto', C = 1, probability = True, verbose = True, random_state = 0)
        elif(self.model_name == 'dt'):
            self.classifier = DecisionTree('gini', 'best', None, 10)
        elif(self.model_name == 'cnn'):
            self.CNN_classifier = CNN(self.input_image_size, self.no_epochs, self.batch_size, self.output_classes, self.learning_rate)
        elif(self.model_name == 'inceptionv3'):
            self.inception_classifier = Inceptionv3(self.output_classes, self.batch_size, self.no_epochs, self.input_image_size)
        #print("Parameters Initialized")
        

    def set_dataset(self, data_X, data_Y, test_X, test_Y, cross_validation_X, cross_validation_Y):
        self.data_X, self.data_Y, self.test_X, self.test_Y, self.cross_validation_X, self.cross_validation_Y = data_X, data_Y, test_X, test_Y, cross_validation_X, cross_validation_Y
        self.no_batches = len(self.data_X) // (self.batch_size)

    def init_model(self):
        if (self.model_name == 'ann'):
            self.NN.create_tf_model("ModelB")

        elif (self.model_name == 'cnn'):
            self.CNN_classifier.initialize_model()

        elif(self.model_name == 'inceptionv3'):
            self.inception_classifier.initialize_model()
        elif (self.model_name == 'svm'):
            pass
        elif(self.model_name == 'dt'):
            pass
        else:
            print("Not implemented yet")

        ##print("Model Initialized")

    def train_model(self):
        if(self.model_name == 'cnn'):
            self.CNN_classifier.train_model(self.data_X, self.data_Y, self.cross_validation_X, self.cross_validation_Y)
        elif(self.model_name == 'inceptionv3'):
            self.inception_classifier.train_model(self.data_X, self.data_Y, self.cross_validation_X, self.cross_validation_Y)
        else:
            self.data_X = self.data_X.reshape(self.data_X.shape[0], -1)
            self.cross_validation_X = self.cross_validation_X.reshape(self.cross_validation_X.shape[0], -1)
            self.test_X = self.test_X.reshape(self.test_X.shape[0], -1)
            
            if (self.model_name == 'svm'):
                #TODO::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::ADD CROSS VALIDATION HERE
                print(self.data_Y.shape)
                self.classifier.fit(self.data_X, self.data_Y)
                print("Training Finished")
                
            elif (self.model_name == 'ann'):
                self.NN.train_model(self.data_X, self.data_Y, self.cross_validation_X, self.cross_validation_Y)
            
            elif(self.model_name == 'dt'):
                self.classifier.train_model(self.data_X, self.data_Y, self.cross_validation_X, self.cross_validation_Y)
            
            else:
                print("Not yet implemented")

    def get_output(self):
        if (self.model_name == 'ann'):
            prediction_probs_train, preds_train, _, acc = self.NN.get_predictions(self.data_X, True,
                                                                               convert_one_hot(self.data_Y,
                                                                                               self.output_classes))
            prediction_probs_train = np.array(prediction_probs_train)
            print(prediction_probs_train.shape)
            # prediction_probs = prediction_probs.reshape((prediction_probs.shape[1], prediction_probs.shape[2]))
            prediction_probs_train = np.argmax(prediction_probs_train, axis=-1)

            prediction_probs_cv, preds_cv, _, acc2 = self.NN.get_predictions(self.cross_validation_X, True,
                                                                          convert_one_hot(self.cross_validation_Y, self.output_classes))
            prediction_probs_cv = np.argmax(np.array(prediction_probs_cv), axis=-1)
            
            prediction_probs_test, preds_test, _ , _ = self.NN.get_predictions(self.test_X, True, convert_one_hot(self.test_Y, self.output_classes))
            prediction_probs_test = np.argmax(np.array(prediction_probs_test), axis=-1)
            
            print("Final Accuracy of Model B on current K-1 folds of the " + self.dataset_name + " Train dataset is :" + str(acc))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Cross Validation dataset is :" + str(acc2))
            #a = input()
            # print(prediction_probs, self.new_data_Y)
            print(np.unique(prediction_probs_train), np.unique(prediction_probs_cv), np.unique(prediction_probs_test))
            return prediction_probs_train, prediction_probs_cv, prediction_probs_test

        elif (self.model_name == 'svm'):
            prediction_probs_train = np.array(self.classifier.predict_proba(self.data_X))
            prediction_probs_cv = np.array(self.classifier.predict_proba(self.cross_validation_X))
            prediction_probs_test = np.array(self.classifier.predict_proba(self.test_X))
            prediction_probs_train = np.argmax(prediction_probs_train, axis=-1)
            prediction_probs_cv = np.argmax(prediction_probs_cv, axis=-1)
            prediction_probs_test = np.argmax(prediction_probs_test, axis = -1)
            # preds_train = self.classifier.predict(self.data_X)
            # preds_cv = self.classifier.predict(self.cross_validation_X)
            print("Final Accuracy of Model B on current K-1 folds of the " + self.dataset_name + " train dataset is :" + str(accuracy_score(self.data_Y, prediction_probs_train)))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Cross Validation dataset is :" + str(accuracy_score(self.cross_validation_Y, prediction_probs_cv)))
            a = input()
            return prediction_probs_train, prediction_probs_cv, prediction_probs_test
        
        elif(self.model_name == 'dt'):
            prediction_probs_train = np.argmax(self.classifier.predict_model(self.data_X), axis = -1)
            prediction_probs_cv = np.argmax(self.classifier.predict_model(self.cross_validation_X), axis = -1)
            prediction_probs_test = np.argmax(self.classifier.predict_model(self.test_X), axis = -1)
            print("Final Accuracy of Model B on current fold of" + self.dataset_name + " CV Dataset is :" + str(accuracy_score(self.cross_validation_Y, prediction_probs_cv)))
            return prediction_probs_train, prediction_probs_cv, prediction_probs_test
            
        elif(self.model_name == 'cnn'):
            prediction_probs_train, preds_train, _, acc = self.CNN_classifier.get_predictions(self.data_X, True, self.data_Y)
            #prediction_probs_train = np.argmax(np.array(prediction_probs_train), axis = -1)
            
            prediction_probs_cv, preds_cv, _, acc2 = self.CNN_classifier.get_predictions(self.cross_validation_X, True, self.cross_validation_Y)
            
            print("Final Accuracy of Model B on current K-1 folds of the " + self.dataset_name + " Train dataset is :" + str(acc))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Cross Validation dataset is :" + str(acc2))
            
            prediction_probs_test, preds_test, _ = self.CNN_classifier.get_predictions(self.test_X, False, self.test_Y)
            
            return preds_train, preds_cv, preds_test
        
        elif(self.model_name == 'inceptionv3'):
            prediction_probs_train, preds_train, _, acc = self.inception_classifier.get_output(self.data_X, True, self.data_Y)
            prediction_probs_cv, preds_cv, _, acc2 = self.inception_classifier.get_output(self.cross_validation_X, True, self.cross_validation_Y)
            
            print("Final Accuracy of Model B on current K-1 folds of the " + self.dataset_name + " Train dataset is :" + str(acc))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Cross Validation dataset is :" + str(acc2))
            
            prediction_probs_test, preds_test, _ = self.inception_classifier.get_output(self.test_X, False, self.test_Y)
            
            return preds_train, preds_cv, preds_test
        else:
            print("Not yet implemented")
            return None, None, None
        
