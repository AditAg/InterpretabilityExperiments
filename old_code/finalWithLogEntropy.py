import os
import tensorflow as tf

from model_classes import ModelA, ModelB
from base_functions import test_kfold_cross_validation, interpretation_diff_frequencies, test_cross_validation, test_kfold_cross_validation_stanford40, test_sentiment_analysis
from all_globals import dataset_name

def main():
    
    #learning_rate = 0.1 #for adadelta optimizer learning rate = 1 works best.
    #batch_size = 128
    
    no_samples_all = (1707, )
    
    for sample in no_samples_all:
        print("No of Samples:", sample)
        #file1 = open("Exp1_No_samples/output_mnist_gini_bestsplit_5.txt", 'a')
        #file1.write("No of samples: " + str(sample) + "\n")
        #file1.close()
        #file1 = open("ensemble_cnn_output_cifar_10.txt", 'a')
        #file1.write("No of samples: " + str(sample) + "\n")
        #file1.close()
        #if(os.path.exists("output.txt")):
        #    file1 = open("output.txt", 'a')
        #else:
        #    file1 = open("output.txt", 'w')
        #file1.write("No of samples: " + str(sample) + "\n")
        no_samples = (sample, 100)
        tf.compat.v1.reset_default_graph()
        
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.85)
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.log_device_placement = True
        session = tf.compat.v1.Session(config=config)
        
        with session as sess:
            if(dataset_name == 'stanford40'):
                test_kfold_cross_validation_stanford40(sess)
            elif(dataset_name == 'sentiment_analysis'):
                test_sentiment_analysis(no_samples, sess)
            else:
                test_kfold_cross_validation(no_samples, sess)
            
        session.close()
        


if __name__ == '__main__':
    main()
