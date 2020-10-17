import os
import pickle
import math
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def data_preprocessing(data, remove_duplicates = True):
    #Remove duplicates
    if(remove_duplicates):
        data = list(set(data))
    
    #remove stopwords
    #extra_stopwords = ["dont", "couldnt", "wont", "shouldnt", "wouldnt", "werent"] and similar cases are also handled
    stopwords = nltk.corpus.stopwords.words('english')
    for i in stopwords:
        if "'" in i:
            new_word = i.replace("'", "")
            #print(i, new_word)
            stopwords.append(new_word)
    
    new_data = []
    for word in data:
        if word not in stopwords:
            new_data.append(word)
    
    #remove punctuation
    #TODO: HYPHENS HAVE TO BE DEALT WITH CAREFULLY,
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for symb in symbols:
        if symb in new_data:
            new_data.remove(symb)
    
    #remove all single length words as they dont give much useful information
    for word in new_data:
        if (len(word)<=1):
            new_data.remove(word)
    
    #perform stemming to convert words to their basic form
    ps = PorterStemmer()
    new_data_2 = []
    for word in new_data:
        new_data_2.append(ps.stem(word))
        #new_data_2.append(word)

    if(remove_duplicates):
        new_data_2 = list(set(new_data_2))
        return new_data_2
    else:
        return new_data_2

def data_preprocessing_function(data):
    #TODO: HYPHENS HAVE TO BE DEALT WITH CAREFULLY,
    new_data = data
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    list_symbols = []
    for symb in symbols:
        list_symbols.append(symb)
    ps = PorterStemmer()
    for index in range(len(data)):
        review = data[index]
        token_sentences = sent_tokenize(review)
        new_sentences = []
        for sent in token_sentences:
            words = word_tokenize(sent)
            new_words = [word for word in words if word not in list_symbols]
            new_words_2 = [word for word in new_words if (len(word) > 1)]
            #new_words = [ps.stem(word) for word in new_words_2]
            new_sent = " ".join(new_words_2)
            new_sentences.append(new_sent)
        new_review = " ".join(new_sentences)
        new_data[index] = new_review
    return new_data
    
def get_bag_words_single_review(set_sentences):
    bag_words = []
    token_sentences = sent_tokenize(set_sentences)
    for sent in token_sentences:
        words = word_tokenize(sent)
        new_words = np.char.lower(words)
        bag_words.extend(new_words)
    return bag_words

def get_bag_of_words(data):
    bag_words = []
    for i in range(data.shape[0]):
        words = get_bag_words_single_review(data[i][1])
        bag_words.extend(words)
    return bag_words

def get_dict(bagOfUniqueWords, data):
    numOfWords = dict.fromkeys(bagOfUniqueWords, 0)
    bagOfWords = get_bag_words_single_review(data[1])
    preprocessed_bagOfWords = data_preprocessing(bagOfWords, remove_duplicates = False)
    for word in preprocessed_bagOfWords:
        if(word in bagOfUniqueWords):
            numOfWords[word] += 1
    return numOfWords, preprocessed_bagOfWords
    
def computeTF(dict_words, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in dict_words.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def computeIDF(documents):
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        #to prevent division by 0, 1 is added to the value here
        idfDict[word] = math.log(N / float(val + 1))
    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        #1 is added to prevent values with idfs = 0 to simply become 0 due to the multiplication below
        tfidf[word] = val * (idfs[word] + 1)
    return tfidf
 
def get_tfidf_dataset(data):
    bag_words_train = get_bag_of_words(data)
    #print(bag_words_train)
    #a = input()
    preprocessed_bag_words_train = data_preprocessing(bag_words_train)
    #print(preprocessed_bag_words_train)
    #a = input()
    #for word in preprocessed_bag_words_train:
    #    if(word in bag_words_train):
    #        print(word, "Yes")
    #    else:
    #        print(word, "No")
    #    a = input()
    final_X = np.empty((data.shape[0], len(preprocessed_bag_words_train)))
    final_Y = data[:, 0]
    final_Y[final_Y == 'Negative'] = 0
    final_Y[final_Y == 'Positive'] = 1
    #print("Hello1")
    #a = input()
    #print(final_train_Y)
    #a = input()
    list_of_term_frequencies = []
    all_document_dicts = []
    for i in range(final_X.shape[0]):
        dict_words, bag_words = get_dict(preprocessed_bag_words_train, data[i])
        #for word, count in dict_words.items():
        #    if(count>0):
        #        print(word, ":", count)
        #a = input()
        tf = computeTF(dict_words, bag_words)
        list_of_term_frequencies.append(tf)
        all_document_dicts.append(dict_words)
    #print("Hello2")
    #a = input()
    #print(all(all_document_dicts[0].get(key, None) == val for key, val in all_document_dicts[1].items()))
    #print(all_document_dicts[0].keys() == all_document_dicts[1].keys())
    #a = input()
    inverse_document_frequency = computeIDF(all_document_dicts)
    #print("Hello3")
    #calculate tf-idf for each sample
    list_tfidf_dicts = []
    for i in range(final_X.shape[0]):
        tfidf = computeTFIDF(list_of_term_frequencies[i], inverse_document_frequency)
        list_tfidf_dicts.append(tfidf)
        
    return final_Y, pd.DataFrame(list_tfidf_dicts)

def get_vectorizer():
    stopwords = nltk.corpus.stopwords.words('english')
    for i in stopwords:
        if "'" in i:
            new_word = i.replace("'", "")
            #print(i, new_word)
            stopwords.append(new_word)
        
    #HOW TO PERFORM STEMMING??????????????????      
    vectorizer = TfidfVectorizer(strip_accents = 'unicode', ngram_range = (1,1), max_df = 0.9, min_df = 1, stop_words = stopwords)
    return vectorizer
    
    
def apply_vectorizer(data, vectorizer, fit = False):
    list_documents = []
    
    final_Y = data[:, 0]
    final_Y[final_Y == 'Negative'] = 0
    final_Y[final_Y == 'Positive'] = 1
        
    for i in range(data.shape[0]):
        list_documents.append(data[i][1])
    
    print(list_documents[0])
    new_list_documents = data_preprocessing_function(list_documents)
    if(fit):
        vectors = vectorizer.fit_transform(new_list_documents)
    else:
        vectors = vectorizer.transform(new_list_documents)
    feature_names = vectorizer.get_feature_names()
    #print(feature_names)
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns = feature_names)
    
    return final_Y.astype('int'), df.values
    
def load_sentiment_dataset():
    file_location = os.path.join(os.getcwd(), 'counterfactually-augmented-data')
    file_location = os.path.join(file_location, 'sentiment')
    file_location = os.path.join(file_location, 'combined')
    file_location = os.path.join(file_location, 'paired')
    df_train = pd.read_csv(os.path.join(file_location, 'train_paired.tsv'), sep = '\t')
    df_cv = pd.read_csv(os.path.join(file_location, 'dev_paired.tsv'), sep = '\t')
    df_test = pd.read_csv(os.path.join(file_location, 'test_paired.tsv'), sep = '\t')
    train_data, cv_data, test_data = df_train.values, df_cv.values, df_test.values
    np.random.shuffle(train_data)
    np.random.shuffle(cv_data)
    np.random.shuffle(test_data)
    
    print(train_data.shape, cv_data.shape, test_data.shape)
    #GET BAG OF WORDS FOR ENTIRE DATA
    total_data = np.concatenate((train_data, cv_data), axis = 0)                 #TOTAL DATA FOR FITTING VECTORIZER
    
    if(not os.path.exists('vectorizer.pk')):
        vectorizer = get_vectorizer()
    else:
        vectorizer_file = open('vectorizer.pk', 'rb')
        vectorizer = pickle.load(vectorizer_file)
    tfidf_total_Y, tfidf_total_X = apply_vectorizer(total_data, vectorizer, fit = False)
    tfidf_train_Y, tfidf_train_X = tfidf_total_Y[:train_data.shape[0]], tfidf_total_X[:train_data.shape[0]]
    #print(tfidf_train_X.shape, tfidf_train_Y.shape)
    #print (tfidf_train_X.head)
    
    tfidf_total_Y, tfidf_total_X = tfidf_total_Y[train_data.shape[0]:], tfidf_total_X[train_data.shape[0]:]
    tfidf_cv_Y, tfidf_cv_X = tfidf_total_Y[:cv_data.shape[0]], tfidf_total_X[:cv_data.shape[0]]
    #print(tfidf_cv_X.shape, tfidf_cv_Y.shape)
    
    tfidf_test_Y, tfidf_test_X = apply_vectorizer(test_data, vectorizer, fit = False)
    #print(tfidf_test_Y.shape, tfidf_test_X.shape)
    
    if(not os.path.exists('vectorizer.pk')):
        with open('vectorizer.pk', 'wb') as fin:
            pickle.dump(vectorizer, fin)
    
    return tfidf_train_X, tfidf_train_Y, tfidf_cv_X, tfidf_cv_Y, tfidf_test_X, tfidf_test_Y
    