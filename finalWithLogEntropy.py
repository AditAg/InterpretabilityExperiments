from model_classes import ModelA, ModelB
from base_functions import test_kfold_cross_validation, interpretation_diff_frequencies, test_cross_validation, test_kfold_cross_validation_stanford40, test_sentiment_analysis
from all_globals import dataset_name

def main():
    
    # list of samples to be considered for training
    # This is used for all experiments where different % of datasets are considered.
    no_samples_all = (1707, )
    
    for sample in no_samples_all:
        print("No of Samples:", sample)
        no_samples = (sample, )
        if(dataset_name == 'stanford40'):
            test_kfold_cross_validation_stanford40()
        elif(dataset_name == 'sentiment_analysis'):
            test_sentiment_analysis()
        else:
            test_kfold_cross_validation(no_samples)

if __name__ == '__main__':
    main()
