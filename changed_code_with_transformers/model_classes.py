import numpy as np
import math
#from h2o4gpu import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ops import convert_one_hot, Neural_Network, DecisionTree, CNN, ensemble_model, fit_model_to_initial_dataset, Inceptionv3
from fns import convert_to_all_classes_array
from ops import get_svm_parameters_sentiment_analysis, get_nb_parameters_sentiment_analysis
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB 
from all_globals import entropy_precision
#from compute_waic import final_computation_WAIC
from test_WAIC_ovo import final_computation_WAIC
from keras.utils import to_categorical
#from thundersvm import SVC
#from sklearn.linear_model import SGDClassifier
#import h2o4gpu as sklearn
#from sklearn import svm

#svm_parameter_list = get_svm_parameters_sentiment_analysis()
svm_parameter_list = {'C' : 1, 'kernel': 'linear'}
nb_parameter_list = {'alpha' : 1}
#nb_parameter_list = get_nb_parameters_sentiment_analysis()
#KNOWN MODEL
class ModelA(object):
    def __init__(self, epochs, batch_size, dataset_name, learning_rate, model_name, is_binarized, is_resized, is_grayscale, feature_size = None, mode = 'original', entropy_mode = 'original'):
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
        self.interpretability_mode = mode
        self.entropy_mode = entropy_mode

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
            self.output_classes = 2 if (self.is_binarized) else 5
        else:
            print("Not Implemented yet")

        if (self.model_name == "svm"):
            if(self.dataset_name == 'sentiment_analysis'):
                #param_list = svm_parameter_list.copy()
                #self.param_list = {'C': 1, 'kernel' : 'linear'}
                #self.param_list = {'C': 0.1, 'kernel' : 'poly', 'gamma' : 10, 'coef0': 2.0, 'degree' : 2}
                self.param_list = {'C': 10, 'kernel' : 'rbf', 'gamma' : 1}
                
                self.param_list['probability'] = True
                self.param_list['random_state'] = 0
                print(self.param_list)
                self.classifier = svm.SVC(C = self.param_list['C'], kernel = self.param_list['kernel'], gamma = self.param_list['gamma'], probability = self.param_list['probability'], random_state = self.param_list['random_state'], decision_function_shape = 'ovo')
            else:
                self.classifier = svm.SVC(C = 1, kernel = 'rbf', gamma = 'auto', probability = True, random_state = 0)
            #self.classifier = SVC(kernel = 'rbf', gamma = 'auto', C = 1, probability = True, verbose = True, random_state = 0)
            self.classifier = fit_model_to_initial_dataset(self.dataset_name, self.classifier, self.model_name, self.is_resized, self.is_grayscale)
         
        elif(self.model_name == "naive_bayes"):
            if(self.dataset_name == 'sentiment_analysis'):
                self.classifier = MultinomialNB(alpha = nb_parameter_list['alpha'])
            else:
                self.classifier = MultinomialNB(alpha = 1.0)
            self.classifier = fit_model_to_initial_dataset(self.dataset_name, self.classifier, self.model_name, self.is_resized, self.is_grayscale) 
             
        elif (self.model_name == "ann"):
            input_size = np.prod(self.input_image_size)
            self.hidden_layers = [256]
            self.NN = Neural_Network(self.no_epochs, self.batch_size, self.learning_rate, input_size,
                                     self.output_classes, self.hidden_layers)
            self.NN.create_tf_model("ModelA")
            #self.fit_model_to_initial_dataset()
            
        elif(self.model_name == "dt"):
            self.classifier = DecisionTree('gini', 'best', None, 5)
            print("Gini_Best_5")
            self.classifier = fit_model_to_initial_dataset(self.dataset_name, self.classifier, "dt", self.is_resized, self.is_grayscale)
            
        elif(self.model_name == "ensemble"):
            input_size = np.prod(self.input_image_size)
            self.classifier = ensemble_model(input_size, self.no_epochs, self.batch_size, self.output_classes, self.learning_rate, "")
            if(self.dataset_name == 'sentiment_analysis'):
                param_list = svm_parameter_list.copy()
                param_list['probability'] = True
                param_list['random_state'] = 0
                param_list['gamma'] = 'scale'
                ensemble_list = [("svm", param_list), ("lr", ), ("dt", "gini", "best", None, 10)]
            else:
                param_list = {'C':0.01, 'kernel':'rbf', 'gamma': 'auto', 'probability':True, 'random_state':0}
                ensemble_list = [("svm", param_list), ("lr", ), ("dt", "gini", "best", None, 10)]
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

    def set_counter_factual_dataset(self, data_X, data_Y, test_X, test_Y, cross_validation_X, cross_validation_Y):
        self.cf_data_X, self.cf_data_Y, self.cf_test_X, self.cf_test_Y, self.cf_cv_X, self.cf_cv_Y = data_X, data_Y, test_X, test_Y, cross_validation_X, cross_validation_Y
    
    def calculate_interpretability(self, probs_model_B_train, probs_model_B_cross_validation, probs_model_B_test, probs_B_train_cf = None, probs_B_cv_cf = None, probs_B_test_cf = None):
        if(self.model_name =='ann' or self.model_name == 'svm' or self.model_name == 'dt' or self.model_name == "ensemble" or self.model_name == "naive_bayes"):
            self.data_X = self.data_X.reshape(self.data_X.shape[0], -1)
            self.cross_validation_X = self.cross_validation_X.reshape(self.cross_validation_X.shape[0], -1)
            self.test_X = self.test_X.reshape(self.test_X.shape[0], -1)
            if(self.interpretability_mode == 'counter_factual'):
                self.cf_data_X = self.cf_data_X.reshape(self.cf_data_X.shape[0], -1)
                self.cf_cv_X = self.cf_cv_X.reshape(self.cf_cv_X.shape[0], -1)
                self.cf_test_X = self.cf_test_X.reshape(self.cf_test_X.shape[0], -1)
            
        
        ##print("The size of the final dataset used for interpretation is: ", self.data_X.shape)
        if(self.interpretability_mode == 'counter_factual'):
            print("Entropy calculated on counter factual data")
            initial_entropy_train = self.calculate_entropy(probs_B_train_cf, 'initial', 0)
            initial_entropy_cv = self.calculate_entropy(probs_B_cv_cf, 'initial', 1)
            initial_entropy_test = self.calculate_entropy(probs_B_test_cf, 'initial', 2)
        elif(self.interpretability_mode == 'original'):
            initial_entropy_train = self.calculate_entropy(probs_model_B_train, 'initial', 0)
            initial_entropy_cv = self.calculate_entropy(probs_model_B_cross_validation, 'initial', 1)
            initial_entropy_test = self.calculate_entropy(probs_model_B_test, 'initial', 2)
        #initial_entropy_train, initial_entropy_test = None, None
        #print(initial_entropy_train, initial_entropy_cv, initial_entropy_test)
        
        #compute WAIC for model A for SVMs
        if(self.model_name == 'naive_bayes'):
            if(self.interpretability_mode == 'counter_factual'):
                if(self.entropy_mode == 'counter_factual'):
                    waic_tr_data_X = self.cf_data_X
                    waic_tr_data_Y = probs_B_train_cf
                    
                elif(self.entropy_mode == 'original'):
                    waic_tr_data_X = self.data_X
                    waic_tr_data_Y = probs_model_B_train

                waic_calc_data_X = self.cf_data_X
                waic_calc_data_Y = probs_B_train_cf
                waic_test_X = self.cf_test_X
                waic_test_Y = probs_B_test_cf
                
            elif(self.interpretability_mode == 'original'):
                waic_tr_data_X = self.data_X
                waic_test_X = self.test_X
                waic_calc_data_X = self.data_X
                waic_calc_data_Y = probs_model_B_train
                waic_tr_data_Y = probs_model_B_train
                waic_test_Y = probs_model_B_test
            params = self.param_list.copy()
            waic_tr_data_Y = np.argmax(waic_tr_data_Y, axis = -1)
            waic_calc_data_Y = np.argmax(waic_calc_data_Y, axis = -1)
            waic_test_Y = np.argmax(waic_test_Y, axis = -1)
            print("Unique Classes in train_data for WAIC", np.unique(waic_tr_data_Y))
            print("Unique Classes in calc_data for WAIC", np.unique(waic_calc_data_Y))
            print("Unique Classes in test_data for WAIC", np.unique(waic_test_Y))
            del params['kernel']
            waic_train_per_class, waic_test_per_class, waic_train_total, waic_test_total = final_computation_WAIC(self.param_list['kernel'], waic_tr_data_X, waic_tr_data_Y, waic_calc_data_X, waic_calc_data_Y, waic_test_X, waic_test_Y, self.classifier, params)
            print("Per class WAIC values: Train: ", waic_train_per_class, " Test: ", waic_test_per_class)
            print("Overall WAIC values: Train: ", waic_train_total," Test: ", waic_test_total)
        
        if(self.interpretability_mode == 'counter_factual'):
            if(self.entropy_mode == 'counter_factual'):
                self.initialize_model(probs_B_train_cf, probs_B_test_cf)
            elif(self.entropy_mode == 'original'):
                self.initialize_model(probs_model_B_train, probs_model_B_cross_validation)
        elif(self.interpretability_mode == 'original'):
            self.initialize_model(probs_model_B_train, probs_model_B_cross_validation)
        
        if(self.interpretability_mode == 'original'):
            final_entropy_train = self.calculate_entropy(probs_model_B_train, 'final', 0)
            final_entropy_cv = self.calculate_entropy(probs_model_B_cross_validation, 'final', 1)
            final_entropy_test = self.calculate_entropy(probs_model_B_test, 'final', 2)
        elif(self.interpretability_mode == 'counter_factual'):
            print("Entropy calculated on counter factual data")
            final_entropy_train = self.calculate_entropy(probs_B_train_cf, 'final', 0)
            final_entropy_cv = self.calculate_entropy(probs_B_cv_cf, 'final', 1)
            final_entropy_test = self.calculate_entropy(probs_B_test_cf, 'final', 2)
        #final_entropy_train, final_entropy_test = None, None
        #a = input()
        print(final_entropy_train, final_entropy_cv, final_entropy_test)
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
        #re-initialize the classifier to original state for SVM and Decision Tree.
        #This is necessary to ensure proper cross-validation.
        if(self.model_name == 'svm' or self.model_name == 'dt' or self.model_name == 'naive_bayes'):
            self.classifier = fit_model_to_initial_dataset(self.dataset_name, self.classifier, self.model_name, self.is_resized, self.is_grayscale)
        elif(self.model_name == "ensemble"):
            self.classifier.fit_model(self.is_resized, self.is_grayscale)
        return interpret_train, interpret_cv, interpret_test

    def initialize_model(self, probs_model_B_train, probs_model_B_cv):
        predictions_model_B_train = np.argmax(probs_model_B_train, axis = -1)
        predictions_model_B_cv = np.argmax(probs_model_B_cv, axis = -1)
        if(self.interpretability_mode == 'original'):
            data_train = self.data_X
            data_cv = self.cross_validation_X
        elif(self.interpretability_mode == 'counter_factual'):
            if(self.entropy_mode == 'counter_factual'):
                data_train = self.cf_data_X
                data_cv = self.cf_cv_X
            elif(self.entropy_mode == 'original'):
                data_train = self.data_X
                data_cv = self.cross_validation_X
            
        if (self.model_name == 'svm' or self.model_name == 'naive_bayes'):
            # self.classifier = SGDClassifier(loss = "hinge")
            print(data_train.shape, predictions_model_B_train.shape)
            #self.classifier.fit(self.data_X[:10000], predictions_model_B[:10000])                      CHANGE
            #TODO::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::ADD CROSS VALIDATION HERE
            self.classifier.fit(data_train, predictions_model_B_train)
            print("Fitted the" + self.model_name + " to the Predictions of Model B")

        elif (self.model_name == 'ann'):
            data_train = data_train.reshape(data_train.shape[0], -1)
            print(data_train.shape, predictions_model_B_train.shape, data_cv.shape)
            self.NN.train_model(data_train, predictions_model_B_train, data_cv, predictions_model_B_cv)
            print("Trained the Neural Network Model A on Predictions of Model B")
        
        elif(self.model_name == 'dt'):
            data_train = data_train.reshape(data_train.shape[0], -1)
            data_cv = data_cv.reshape(data_cv.shape[0], -1)
            #preds2 = np.argmax(self.classifier.predict_model(data_train), axis = -1)
            #print("Final Accuracy of Model A on MNIST Train Dataset before interpretation is :" + str(metrics.accuracy_score(self.data_Y, preds2)))
            self.classifier.train_model(data_train, predictions_model_B_train, data_cv, predictions_model_B_cv)
            #print("Fitted the Decision Tree to the predictions of model B")
            #preds = np.argmax(self.classifier.predict_model(data_cv), axis = -1)
            #print("Final Accuracy of Model A on MNIST CV Dataset is :" + str(metrics.accuracy_score(self.cross_validation_Y, preds)))
            #preds2 = np.argmax(self.classifier.predict_model(data_train), axis = -1)
            #print("Final Accuracy of Model A on MNIST CV Dataset is :" + str(metrics.accuracy_score(self.data_Y, preds2)))
        
        elif(self.model_name == "ensemble"): #cannot have CNN here, as the data_X is being flattened out here, for use by other models
            data_train = data_train.reshape(data_train.shape[0], -1)
            data_cv = data_cv.reshape(data_cv.shape[0], -1)
            self.classifier.train_ensemble(data_train, predictions_model_B_train, data_cv, predictions_model_B_cv)
        elif(self.model_name == "cnn"):
            self.CNN_classifier.train_model(data_train, predictions_model_B_train, data_cv, predictions_model_B_cv)
        elif(self.model_name == "inceptionv3"):
            self.inception_classifier.train_model(data_train, predictions_model_B_train, data_cv, predictions_model_B_cv)
        else:
            print("Not implemented yet")

    def calculate_entropy(self, probs_B, name, split):
        # prob = np.array(self.classifier.decision_function(self.data_X))
        # prob_B_indexes = np.argmax(predictions_model_B, axis = -1)
        preds = np.argmax(probs_B, axis = -1)
        if(split == 0):
            if(self.interpretability_mode == 'original'):
                data = self.data_X
                output = self.data_Y
            elif(self.interpretability_mode == 'counter_factual'):
                data = self.cf_data_X
                output = self.cf_data_Y
            #print("Train split")
        elif(split == 1):
            if(self.interpretability_mode == 'original'):
                data = self.cross_validation_X
                output = self.cross_validation_Y
            elif(self.interpretability_mode == 'counter_factual'):
                data = self.cf_cv_X
                output = self.cf_cv_Y
            #print("Cross Validation split ")
        elif(split == 2):
            if(self.interpretability_mode == 'original'):
                data = self.test_X
                output = self.test_Y
            elif(self.interpretability_mode == 'counter_factual'):
                data = self.cf_test_X
                output = self.cf_test_Y
            #print("Test split")
        else:
            #print("Invalid Split Value")
            return None
        if (self.model_name == 'svm' or self.model_name == 'naive_bayes'):
            #probs_train = np.array(self.classifier.predict_proba(self.data_X[:10000]))                CHANGE
            probs2 = np.array(self.classifier.predict_proba(data))
            print(probs2.shape)
            probs = convert_to_all_classes_array(probs2, self.classifier.classes_, self.output_classes)
            if(probs2.shape[-1] == self.output_classes):
                assert(probs2.all() == probs.all())
            #print(probs)
            #a = input()
            ######CHANGE MADE IN LINE BELOW, CONFIRM : Earlier: probs2, Now: probs
            categorical_outputs = to_categorical(output)
            print("Accuracy of Model A on the current split of the dataset is : ", accuracy_score(np.argmax(categorical_outputs, axis = -1), np.argmax(probs, axis = -1)))
            
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
        print("Accuracy of Model A on the current split of the predictions of Model B of the dataset is: ", accuracy_score(preds, prob_A_indexes))
        ##print("Classes predicted by the model A: ", np.unique(prob_A_indexes))
        ##print(probs_train.shape, preds_train.shape, prob_A_indexes.shape)
        
        total_diff = 0.0
        if(self.dataset_name == "sentiment_analysis"):
            count_equal, count_unequal = 0, 0
            for i in range(probs.shape[0]):
                if(preds[i] == prob_A_indexes[i]):
                    count_equal += 1
                else:
                    count_unequal += 1
                diff1 = abs(probs[i][preds[i]] - probs_B[i][preds[i]])
                val1 = 0.0
                if(diff1 != 1.0):
                    val1 = -1.0  * (math.log2(1.0 - diff1))
                else:
                    max_val = -1.0 * math.ceil(math.log2(entropy_precision))
                    val1 = max_val
                total_diff += val1
            total_diff = (total_diff) / (probs.shape[0])
            prob_equal = (count_equal * 1.0)/(probs.shape[0])
            prob_unequal = (count_unequal * 1.0)/(probs.shape[0])
            #total_diff = -1.0 * (prob_equal) * math.log2(prob_equal)
            if(prob_equal == 0 or prob_unequal == 0):
                total_diff = 0
            else:
                #total_diff = -1.0 * math.log2(prob_equal)
                total_diff = (-1.0 * (prob_equal) * math.log2(prob_equal)) + (-1.0 * (prob_unequal) * (math.log2(prob_unequal)))
            
        else:
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
                print("For Model A " + str(self.model_name) + " and Model B, the final predictions on this split of the dataset are same")
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
    def __init__(self, epochs, batch_size, dataset_name, learning_rate, model_name, is_binarized, is_resized, is_grayscale, feature_size = None):
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
            self.output_classes = 2 if (self.is_binarized) else 5
        else:
            print("Not Implemented yet")

        if (self.model_name == 'ann'):
            input_size = np.prod(self.input_image_size)
            #self.hidden_layers = [512, 256, 128, 64]
            self.hidden_layers = [512, 128]
            print(self.hidden_layers)
            self.NN = Neural_Network(self.no_epochs, self.batch_size, self.learning_rate, input_size,
                                     self.output_classes, self.hidden_layers, mc_dropout =  False, dropout_rate = None)

        elif (self.model_name == 'svm'):
            if(self.dataset_name == 'sentiment_analysis'):
                param_list = svm_parameter_list.copy()
                param_list['probability'] = True
                param_list['random_state'] = 0
                self.classifier = svm.SVC(C = param_list['C'], kernel = param_list['kernel'], probability = param_list['probability'], random_state = param_list['random_state'])
            else:
                self.classifier = svm.SVC(C = 1, kernel = 'rbf', gamma = 'auto', probability = True, random_state = 0)
            #self.classifier = SVC(kernel = 'rbf', gamma = 'auto', C = 1, probability = True, verbose = True, random_state = 0)
        
        elif(self.model_name == "naive_bayes"):
            if(self.dataset_name == 'sentiment_analysis'):
                print(nb_parameter_list)
                self.classifier = MultinomialNB(alpha = nb_parameter_list['alpha'])
            else:
                self.classifier = MultinomialNB(alpha = 1.0)
                
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
        
    def set_counter_factual_dataset(self, data_X, data_Y, test_X, test_Y, cross_validation_X, cross_validation_Y):
        self.cf_data_X, self.cf_data_Y, self.cf_test_X, self.cf_test_Y, self.cf_cv_X, self.cf_cv_Y = data_X, data_Y, test_X, test_Y, cross_validation_X, cross_validation_Y
        
    def init_model(self):
        if (self.model_name == 'ann'):
            self.NN.create_tf_model("ModelB")

        elif (self.model_name == 'cnn'):
            self.CNN_classifier.initialize_model()

        elif(self.model_name == 'inceptionv3'):
            self.inception_classifier.initialize_model()
        elif (self.model_name == "svm" or self.model_name == "dt" or self.model_name == "naive_bayes"):
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
            
            if (self.model_name == 'svm' or self.model_name == 'naive_bayes'):
                #TODO::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::ADD CROSS VALIDATION HERE
                print(self.data_Y.shape)
                print("Classes in data used for training")
                print(np.unique(self.data_Y))
                classes_ = np.unique(self.data_Y)
                for class_ in classes_:
                    indices = (self.data_Y == class_).nonzero()
                    print("Class: ", class_, "No of samples: ", np.array(indices).shape)
                print("Classes in data used for test")
                print(np.unique(self.test_Y))
                classes_ = np.unique(self.test_Y)
                for class_ in classes_:
                    indices = (self.test_Y == class_).nonzero()
                    print("Class: ", class_, "No of samples: ", np.array(indices).shape)
                    
                self.classifier.fit(self.data_X, self.data_Y)
                print("Training Finished")
                
            elif (self.model_name == 'ann'):
                self.NN.train_model(self.data_X, self.data_Y, self.cross_validation_X, self.cross_validation_Y)
            
            elif(self.model_name == 'dt'):
                self.classifier.train_model(self.data_X, self.data_Y, self.cross_validation_X, self.cross_validation_Y)
            
            else:
                print("Not yet implemented")

    def get_output(self, mode = 'original'):
        if(mode == 'original'):
            tr_X, tr_Y = self.data_X, self.data_Y
            cv_X, cv_Y = self.cross_validation_X, self.cross_validation_Y
            te_X, te_Y = self.test_X, self.test_Y
        elif(mode == 'counter_factual'):
            tr_X, tr_Y = self.cf_data_X, self.cf_data_Y
            cv_X, cv_Y = self.cf_cv_X, self.cf_cv_Y
            te_X, te_Y = self.cf_test_X, self.cf_test_Y
            
        if (self.model_name == 'ann'):
            prediction_probs_train, preds_train, _, acc = self.NN.get_predictions(tr_X, True, convert_one_hot(tr_Y, self.output_classes))
            prediction_probs_train = np.array(prediction_probs_train)
            print(prediction_probs_train.shape)
            #prediction_probs_train = np.argmax(prediction_probs_train, axis=-1)

            prediction_probs_cv, preds_cv, _, acc2 = self.NN.get_predictions(cv_X, True, convert_one_hot(cv_Y, self.output_classes))
            prediction_probs_cv = np.array(prediction_probs_cv)
            
            prediction_probs_test, preds_test, _ , acc3 = self.NN.get_predictions(te_X, True, convert_one_hot(te_Y, self.output_classes))
            prediction_probs_test = np.array(prediction_probs_test)
            #prediction_probs_test = np.argmax(np.array(prediction_probs_test), axis=-1)
            
            print("Final Accuracy of Model B on current K-1 folds of the " + self.dataset_name + " Train " + mode + " dataset is :" + str(acc))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Cross Validation " + mode + " dataset is :" + str(acc2))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Test " + mode + " dataset is :" + str(acc3))
            return prediction_probs_train, prediction_probs_cv, prediction_probs_test

        elif (self.model_name == 'svm' or self.model_name == 'naive_bayes'):
            prediction_probs_train = np.array(self.classifier.predict_proba(tr_X))
            prediction_probs_cv = np.array(self.classifier.predict_proba(cv_X))
            prediction_probs_test = np.array(self.classifier.predict_proba(te_X))
            print("Final Accuracy of Model B on current K-1 folds of the " + self.dataset_name + " train " + mode + " dataset is :" + str(accuracy_score(tr_Y, np.argmax(prediction_probs_train, axis = -1))))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Cross Validation " + mode + " dataset is :" + str(accuracy_score(cv_Y, np.argmax(prediction_probs_cv, axis = -1))))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Test " + mode + " dataset is :" + str(accuracy_score(te_Y, np.argmax(prediction_probs_test, axis = -1))))
            print("Unique classes in predictions of model B on Test : ", np.unique(np.argmax(prediction_probs_test, axis = -1)))
            return prediction_probs_train, prediction_probs_cv, prediction_probs_test
        
        elif(self.model_name == 'dt'):
            #prediction_probs_train = np.argmax(self.classifier.predict_model(self.data_X), axis = -1)
            #prediction_probs_cv = np.argmax(self.classifier.predict_model(self.cross_validation_X), axis = -1)
            #prediction_probs_test = np.argmax(self.classifier.predict_model(self.test_X), axis = -1)
            prediction_probs_train = self.classifier.predict_model(tr_X)
            prediction_probs_cv = self.classifier.predict_model(cv_X)
            prediction_probs_test = self.classifier.predict_model(te_X)
            
            print("Final Accuracy of Model B on current fold of" + self.dataset_name + " CV " + mode + " Dataset is :" + str(accuracy_score(cv_Y, np.argmax(prediction_probs_cv, axis = -1))))
            return prediction_probs_train, prediction_probs_cv, prediction_probs_test
            
        elif(self.model_name == 'cnn'):
            prediction_probs_train, preds_train, _, acc = self.CNN_classifier.get_predictions(tr_X, True, tr_Y)
            #prediction_probs_train = np.argmax(np.array(prediction_probs_train), axis = -1)
            
            prediction_probs_cv, preds_cv, _, acc2 = self.CNN_classifier.get_predictions(cv_X, True, cv_Y)
            
            print("Final Accuracy of Model B on current K-1 folds of the " + self.dataset_name + " Train " + mode + " dataset is :" + str(acc))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Cross Validation " + mode + " dataset is :" + str(acc2))
            
            prediction_probs_test, preds_test, _ = self.CNN_classifier.get_predictions(te_X, False, te_Y)
            
            #return preds_train, preds_cv, preds_test
            return prediction_probs_train, prediction_probs_cv, prediction_probs_test
        
        elif(self.model_name == 'inceptionv3'):
            prediction_probs_train, preds_train, _, acc = self.inception_classifier.get_output(tr_X, True, tr_Y)
            prediction_probs_cv, preds_cv, _, acc2 = self.inception_classifier.get_output(cv_X, True, cv_Y)
            
            print("Final Accuracy of Model B on current K-1 folds of the " + self.dataset_name + " Train " + mode + " dataset is :" + str(acc))
            print("Final Accuracy of Model B on the current fold of the " + self.dataset_name + " Cross Validation " + mode + " dataset is :" + str(acc2))
            
            prediction_probs_test, preds_test, _ = self.inception_classifier.get_output(te_X, False, te_Y)
            
            #return preds_train, preds_cv, preds_test
            return prediction_probs_train, prediction_probs_cv, prediction_probs_test
        else:
            print("Not yet implemented")
            return None, None, None
        
