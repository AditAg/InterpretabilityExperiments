from base_functions import test_kfold_cross_validation, test_kfold_cross_validation_stanford40, test_sentiment_analysis
from all_globals import dataset_name

def main():
    
    #learning_rate = 0.1 #for adadelta optimizer learning rate = 1 works best.
    #batch_size = 128
    
    no_samples_all = (1707, )
    
    for sample in no_samples_all:
        print("No of Samples:", sample)
        no_samples = (sample, 100)
        if(dataset_name == 'stanford40'):
            test_kfold_cross_validation_stanford40()
        elif(dataset_name == 'sentiment_analysis'):
            test_sentiment_analysis()
        else:
            test_kfold_cross_validation(no_samples)

if __name__ == '__main__':
    main()
