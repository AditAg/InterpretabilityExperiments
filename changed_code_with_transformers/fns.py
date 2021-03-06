import pickle
import os
import numpy as np
import cv2
import gzip
import pandas as pd
from sklearn.preprocessing import StandardScaler
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import re
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import scipy.io
from xml.etree import ElementTree as ET
from sklearn.model_selection import train_test_split
from save_bert_embeddings import save_embeddings

def unpickle(file):
 '''Load byte data from file'''
 with open(file, 'rb') as f:
  data = pickle.load(f, encoding='latin-1')
  return data

def convert_to_grayscale(array, shape):
    new_array = np.zeros((array.shape[0],) + shape)
    for i in range(array.shape[0]):
        new_array[i] = cv2.cvtColor(array[i].astype('float32'), cv2.COLOR_RGB2GRAY)
        
    return new_array

def scale_input(x):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x
    
def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val)/(max_val - min_val)
    return x
        
def load_cifar10_dataset(grayscale, is_resized):
    data_dir = 'data/cifar-10-batches-py'
    train_data = None
    train_labels = []
    
    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic['data']
        else:
            train_data = np.vstack((train_data, data_dic['data']))
        train_labels += data_dic['labels']
    
    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic['data']
    test_labels = test_data_dic['labels']
    
    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    train_data = np.rollaxis(train_data, 1, 4)
    
    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    test_data = np.rollaxis(test_data, 1, 4)
    
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    #cv2.imshow('img', train_data[0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    if(grayscale):
        train_data = convert_to_grayscale(train_data, (32, 32))
        test_data = convert_to_grayscale(test_data, (32, 32))
    #print (train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
    #print ("Unique classes in CIFAR-10 dataset are: ", np.unique(train_labels))
    #train_data = train_data.reshape((len(train_data)))
    train_data, test_data = train_data.astype('float32'), test_data.astype('float32')
    #train_data = train_data/255.0
    #test_data = test_data/255.0
    for i in range(train_data.shape[0]):
        train_data[i] = normalize(train_data[i])
        #train_data[i] = scale_input(train_data[i])
    for i in range(test_data.shape[0]):
        test_data[i] = normalize(test_data[i])
        #test_data[i] = scale_input(test_data[i])
        
    if(is_resized):
        train_data, test_data = resize_images(train_data, test_data, (28, 28))
        
    return train_data, train_labels, test_data, test_labels

def make_binarized(y_train, y_test):
    init_shape = y_train.shape[0]
    Y = np.concatenate((y_train, y_test), axis = 0).astype(np.int)
    Y_binarized = convert_to_binary_classes(Y)
    print(Y_binarized.shape)
    y_train = Y_binarized[:init_shape]
    y_test = Y_binarized[init_shape:]
    return y_train, y_test

def shuffle_dataset(x_train, y_train, x_test, y_test):
    seed = 333
    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)
    seed = 111
    np.random.seed(seed)
    np.random.shuffle(x_test)
    np.random.seed(seed)
    np.random.shuffle(y_test)
    return x_train, y_train, x_test, y_test

def resize_images(x_train, x_test, new_shape):
    x_train = np.array(x_train, dtype = 'uint8')
    #plt.imshow(x_train[0].reshape((28, 28)), cmap = 'gray')
    #plt.show()
    new_x_train = np.zeros((x_train.shape[0], ) + new_shape)
    for i in range(x_train.shape[0]):
        new_img = cv2.resize(x_train[i], dsize = new_shape, interpolation = cv2.INTER_CUBIC)
        new_x_train[i] = new_img
        
    x_test = np.array(x_test, dtype = 'uint8')
    new_x_test = np.zeros((x_test.shape[0],) + new_shape)
    for i in range(x_test.shape[0]):
        new_img = cv2.resize(x_test[i], dsize = new_shape, interpolation = cv2.INTER_CUBIC)
        new_x_test[i] = new_img
        
    new_x_train = new_x_train.reshape(new_x_train.shape + (1,))
    new_x_test = new_x_test.reshape(new_x_test.shape + (1,))
    return new_x_train, new_x_test
    
def extract_data(filename, data_len, data_size, head_size):
    with gzip.open(filename) as bytestream:
        bytestream.read(head_size)
        buf = bytestream.read(data_size * data_len)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
    return data


def load_mnist_dataset(is_binarized, is_resized):
    file_path = os.path.join('data', 'Original dataset')
    x_train = extract_data(os.path.join(file_path, 'train-images-idx3-ubyte.gz'), 60000, 28 * 28, 16)
    y_train = extract_data(os.path.join(file_path, 'train-labels-idx1-ubyte.gz'), 60000, 1, 8)
    x_test = extract_data(os.path.join(file_path, 't10k-images-idx3-ubyte.gz'), 10000, 28 * 28, 16)
    y_test = extract_data(os.path.join(file_path, 't10k-labels-idx1-ubyte.gz'), 10000, 1, 8)
    x_train, x_test = x_train / 255, x_test / 255
    x_train = (x_train > 0.5).astype(np.int_)
    x_test = (x_test > 0.5).astype(np.int_)
    x_train = x_train.reshape((60000, 28, 28, 1))
    y_train = y_train.reshape((60000))
    x_test = x_test.reshape((10000, 28, 28, 1))
    y_test = y_test.reshape((10000))
    # plt.imshow(x_train[0].reshape((28, 28)), cmap = 'gray')
    # plt.show()
    if (is_binarized):
        y_train, y_test = make_binarized(y_train, y_test)
    #print(y_train.shape, y_test.shape, type(y_train), type(y_test))
    
    x_train, y_train, x_test, y_test = shuffle_dataset(x_train, y_train, x_test, y_test)
    
    if(is_resized):
        new_x_train, new_x_test = resize_images(x_train, x_test, (10, 10))
        #plt.imshow(new_x_train[0].reshape((10, 10)), cmap = 'gray')
        #plt.show()
        return new_x_train, y_train, new_x_test, y_test
    return x_train, y_train, x_test, y_test
    # cv2.imshow('image', x_train[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def load_fashionmnist(is_binarized, is_resized):
    ((x_train, y_train), (x_test, y_test)) = fashion_mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    x_train = x_train.reshape((60000, 28, 28, 1))
    y_train = y_train.reshape((60000))
    x_test = x_test.reshape((10000, 28, 28, 1))
    y_test = y_test.reshape((10000))
    if(is_binarized):
        y_train, y_test = make_binarized(y_train, y_test)
    
    x_train, y_train, x_test, y_test = shuffle_dataset(x_train, y_train, x_test, y_test)
    
    if(is_resized):
        x_train, x_test = resize_images(x_train, x_test, (10, 10))
    
    return x_train, y_train, x_test, y_test


#Makes the maximally present class as 1 and the rest as 0.
def convert_to_binary_classes(labels_array):
    my_dict = {}
    for label in labels_array:
        if label not in my_dict.keys():
            my_dict[label] = 1
        else:
            my_dict[label] += 1
    max_label = max(my_dict, key=my_dict.get)
    new_labels_array = labels_array
    for i in range(len(labels_array)):
        if (labels_array[i] == max_label):
            new_labels_array[i] = 1
        else:
            new_labels_array[i] = 0
    return new_labels_array

def find_average(l):
    return sum(l)/len(l)

def convert_to_all_classes_array(probs, classes, final_no_classes):
    assert(probs.shape[-1] == len(classes))
    new_probs = np.zeros((probs.shape[0], final_no_classes), float)
    for i in range(probs.shape[0]):
        for j in range(len(classes)):
            new_probs[i][classes[j]] = probs[i][j]
    
    return new_probs

def load_svhn(grayscale):
    path2 = os.path.join('data', 'SVHN')
    path2 = os.path.join(path2, 'extra_32x32.mat')
    train_data = scipy.io.loadmat(path2)
    #print("Loaded SVHN dataset")
    X = train_data['X']
    X = np.rollaxis(X, 3)
    Y = train_data['y']
    Y[Y==10] = 0      #SVHN has classes from 1 to 10 where 10 is for class '0'
    X = X[:10000]
    Y = Y[:10000]
    if(grayscale):
        X = convert_to_grayscale(X, (32, 32))
    return X, Y


def find_fourier_tranform(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    plt.subplot(121), plt.imshow(image, cmap = 'gray')
    plt.title('Input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    return magnitude_spectrum

def get_high_frequency_image(image, fshift, width):
    rows, cols = image.shape
    crow, ccol = rows/2, cols/2
    crow, ccol = int(crow), int(ccol)
    high_pass_image = fshift
    high_pass_image[(crow - width):(crow + width), (ccol - width):(ccol + width)] = 0
    high_pass_f_ishift = np.fft.ifftshift(high_pass_image)
    high_pass_img = np.fft.ifft2(high_pass_f_ishift)
    high_pass_img = np.abs(high_pass_img)
    
    plt.subplot(131), plt.imshow(image, cmap = 'gray')
    plt.title('Input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(high_pass_img, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(high_pass_img)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
    plt.show()
    return high_pass_img

def get_low_frequency_image(image, fshift, width):
    rows, cols = image.shape
    crow, ccol = rows/2, cols/2
    crow, ccol = int(crow), int(ccol)
    new_image = np.zeros((rows, cols))
    x1, x2, y1, y2 = (crow - width), (crow + width), (ccol - width), (ccol + width)
    new_image[x1:x2, y1:y2] = fshift[x1:x2, y1:y2]
    low_pass_f_ishift = np.fft.ifftshift(new_image)
    low_pass_img = np.fft.ifft2(low_pass_f_ishift)
    low_pass_img = np.abs(low_pass_img)
    
    plt.subplot(131), plt.imshow(image, cmap = 'gray')
    plt.title('Input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(low_pass_img, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(low_pass_img)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
    plt.show()
    return low_pass_img
    
   
def get_frequency_components_dataset(images, width):
    high_frequency_components = images
    low_frequency_components = images
    for i in range(images.shape[0]):
        magnitude_spectrum = find_fourier_tranform(images[i])
        high_frequency_components[i] = get_high_frequency_image(images[i], magnitude_spectrum, width)
        low_frequency_components[i] = get_low_frequency_image(images[i], magnitude_spectrum, width)
    return high_frequency_components, low_frequency_components

def parseXML(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    for child in root:
        if(child.tag == 'object'):
            y_label = child.find('action').text
            xmax = child.find('./bndbox/xmax').text
            xmin = child.find('./bndbox/xmin').text
            ymax = child.find('./bndbox/ymax').text
            ymin = child.find('./bndbox/ymin').text
    #print(y_label, "xmin = ", xmin, "xmax = ", xmax, "ymin = ", ymin, "ymax = ", ymax)
    return y_label, int(xmin), int(xmax), int(ymin), int(ymax)
    
def get_annotation_file_name_from_img(img_name):
    file_path = os.path.join("datasets", "Stanford40")
    annotations_file_path = os.path.join(file_path, "XMLAnnotations")
    img_parts = img_name.split('.')
    img_name_parts = img_parts[0].split('_')
    length = len(img_name_parts)
    #print(img_name_parts)
    if(len(img_name_parts[length - 1]) == 1):
        img_name_parts = img_name_parts[:-1]
    img_name_new = "_"
    img_name_new = img_name_new.join(img_name_parts)
    #img_name_new = img_name_new + "." + img_parts[1]
    #print(img_name_new)
    annotation_file = os.path.join(annotations_file_path, img_name_new) + ".xml"
    #print(annotation_file)
    return parseXML(annotation_file)
    
def get_all_classes(path):
    dict_classes = {}
    i = 0
    for files in os.walk(path):
        for file in files[2]:
            img_name = file.split('.')[0]
            class_name = img_name.split('_')[:-1]
            class_separator = "_"
            class_name = class_separator.join(class_name)
            if(class_name not in dict_classes.keys()):
                dict_classes[class_name] = i
                i = i + 1
    return dict_classes

#TODO: Use the train test split files provided by the Stanford 40 authors  
def load_stanford40_dataset():
    data_X = []
    data_Y = []
    data_X_2 = []
    file_path = os.path.join("datasets", "Stanford40")
    images_file_path_1 = os.path.join(file_path, "JPEGImages")
    images_file_path_2 = os.path.join(file_path, "Stanford40ImagesCropped")
    dict_classes = get_all_classes(images_file_path_1)
    print(dict_classes)
    for i in os.walk(images_file_path_1):
        no_images = 1
        for file in i[2]:
            if(no_images > 1000):
                break
            no_images = no_images + 1
            img = cv2.imread(os.path.join(images_file_path_1, file))
            #img_name = file.split('.')[0]
            self_cropped_image = cv2.imread(os.path.join(images_file_path_2, file))
            new_self_cropped_image = cv2.resize(self_cropped_image, (200, 200))
            #data_X_2.append(new_self_cropped_image)
            ylabel, xmin, xmax, ymin, ymax = get_annotation_file_name_from_img(file)
            y = dict_classes[ylabel]
            height = img.shape[0] - (ymax - ymin)
            if(height <= 10):
            	height = int(img.shape[0]/2)
            width = img.shape[1] - (xmax - xmin)
            if(width <=10):
            	width = int(img.shape[1]/2)
            cropped_image = img[ymin:ymax, xmin:xmax, :]
            cropped_image_2 = img[:(ymax-ymin), :(xmax-xmin),:]
            cropped_image_3 = img[height:, width:, :]
            new_image = cv2.resize(img, (200, 200), interpolation = cv2.INTER_AREA)
            new_cropped_image = cv2.resize(cropped_image, (200, 200))
            new_cropped_image_2 = cv2.resize(cropped_image_2, (200, 200))
            new_cropped_image_3 = cv2.resize(cropped_image_3, (200, 200))
            #print(img.shape)
            #print(cropped_image.shape)
            #print(cropped_image_2.shape)
            print(new_image.shape, new_cropped_image.shape)
            print(new_image.dtype)
            #print(new_image[0][0])
            x = np.ones((10, 200, 3)).astype(np.uint8)*255
            
            #vertic_concat_1 = np.concatenate((new_cropped_image_2, new_cropped_image), axis = 0)
            vertic_concat_1 = np.concatenate((new_cropped_image_2, x), axis = 0)
            vertic_concat_1 = np.concatenate((vertic_concat_1, new_cropped_image), axis = 0)
            
            #vertic_concat_2 = np.concatenate((new_image, new_cropped_image_3), axis = 0)
            vertic_concat_2 = np.concatenate((new_image, x), axis = 0)
            vertic_concat_2 = np.concatenate((vertic_concat_2, new_cropped_image_3), axis = 0)
            
            #horiz_concat = np.concatenate((vertic_concat_1, vertic_concat_2), axis = 1)
            horiz_concat = np.concatenate((vertic_concat_1, np.ones((vertic_concat_1.shape[0], 10, vertic_concat_1.shape[2])).astype(np.uint8)*255), axis = 1)
            horiz_concat = np.concatenate((horiz_concat, vertic_concat_2), axis = 1)
            print(y); 
            cv2.imshow('image_1', x);
            cv2.imshow('image', new_image); 
            cv2.imshow('cropped_image', new_cropped_image)
            cv2.imshow('cropped_image_2', new_cropped_image_2)
            cv2.imshow('cropped_image_3', new_cropped_image_3)
            cv2.imshow('concat_image', horiz_concat); cv2.waitKey(0); cv2.destroyAllWindows()
            a = input()
            data_X.append(new_image)
            data_X_2.append(new_cropped_image)
            data_Y.append(y)
            #a = input()
    nump_data_X = np.array(data_X)
    nump_data_X_2 = np.array(data_X_2)
    nump_data_Y = np.array(data_Y)
    new_data = np.hstack((nump_data_X, nump_data_X_2))
    print(new_data.shape)
    data_X, test_X, data_Y, test_Y = train_test_split(new_data, nump_data_Y, test_size = 0.2, random_state = 10)
    data_X, data_Y, test_X, test_Y = shuffle_dataset(data_X, data_Y, test_X, test_Y)
    data_X, data_X_2 = np.split(data_X, 2, axis = 1)
    test_X, test_X_2 = np.split(test_X, 2, axis = 1)
    print(data_X.shape, data_X_2.shape, test_X.shape, test_X_2.shape, data_Y.shape, test_Y.shape)
    
    #for i in range(test_X.shape[0]):
    #    print(test_Y[i]);
    #    cv2.imshow('image', test_X[i]); 
    #    cv2.imshow('cropped_image', test_X_2[i]); 
    #    cv2.waitKey(0); 
    #    cv2.destroyAllWindows()
    #    a = input()       
    return data_X, data_Y, test_X, test_Y, data_X_2, test_X_2

#t_X, t_Y, te_X, te_Y, t_X_2, te_X_2 = load_stanford40_dataset()
#print(t_X.shape, t_Y.shape, te_X.shape, te_Y.shape, t_X_2.shape, te_X_2.shape)
#cv2.imshow('image', t_X[0])
#cv2.imshow('cropped_image', t_X_2[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()      
  


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

def convert_to_int(data, classification_type):
    if(classification_type == 'binary'):
        data[:, 0][data[:, 0] == 'Negative'] = 0
        data[:, 0][data[:, 0] == 'Positive'] = 1
    else:
        data[:, 1][data[:, 1] == 'Very negative'] = 0
        data[:, 1][data[:, 1] == 'Negative'] = 1
        data[:, 1][data[:, 1] == 'Neutral'] = 2
        data[:, 1][data[:, 1] == 'Positive'] = 3
        data[:, 1][data[:, 1] == 'Very positive'] = 4
    
    return data

def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)

def apply_bert_sentence_embeddings(train_reviews, cv_reviews, test_reviews):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings_train = model.encode(train_reviews)
    sentence_embeddings_cv = model.encode(cv_reviews)
    sentence_embeddings_test = model.encode(test_reviews)
    
    sentence_embeddings_train = np.array(sentence_embeddings_train)
    sentence_embeddings_cv = np.array(sentence_embeddings_cv)
    sentence_embeddings_test = np.array(sentence_embeddings_test)
    
    return sentence_embeddings_train, sentence_embeddings_cv, sentence_embeddings_test
    
def load_sentence_embeddings(mode = 'original', classification_type = 'binary'):
    file_location = os.path.join(os.getcwd(), '..')
    file_location = os.path.join(file_location, 'counterfactually-augmented-data')
    file_location = os.path.join(file_location, 'sentiment')
    if(mode == 'original'):
        file_location = os.path.join(file_location, 'orig')
    elif(mode == 'counter_factual'):
        file_location = os.path.join(file_location, 'new')
    df_train = pd.read_csv(os.path.join(file_location, 'new_train.tsv'), sep = '\t')
    df_cv = pd.read_csv(os.path.join(file_location, 'new_dev.tsv'), sep = '\t')
    df_test = pd.read_csv(os.path.join(file_location, 'new_test.tsv'), sep = '\t')
    train_data, cv_data, test_data = df_train.values, df_cv.values, df_test.values
    np.random.shuffle(train_data)
    np.random.shuffle(cv_data)
    np.random.shuffle(test_data)
    
    train_data = convert_to_int(train_data, classification_type)
    cv_data = convert_to_int(cv_data, classification_type)
    test_data = convert_to_int(test_data, classification_type)
    
    train_reviews = []
    sentences = list(train_data[:, 2])
    for sen in sentences:
        train_reviews.append(preprocess_text(sen))
        
    cv_reviews = []
    sentences = list(cv_data[:, 2])
    for sen in sentences:
        cv_reviews.append(preprocess_text(sen))
    
    test_reviews = []
    sentences = list(test_data[:, 2])
    for sen in sentences:
        test_reviews.append(preprocess_text(sen))
    
    if(classification_type == 'binary'):
        train_y = train_data[:, 0]
        cv_y = cv_data[:, 0]
        test_y = test_data[:, 0]
    else:
        train_y = train_data[:, 1]
        cv_y = cv_data[:, 1]
        test_y = test_data[:, 1]
    
    #Apply BERT Sentence Embeddings
    sentence_embeddings_train, sentence_embeddings_cv, sentence_embeddings_test = apply_bert_sentence_embeddings(train_reviews, cv_reviews, test_reviews)
    
    #TODO: Apply Universal Sentence Encoder
    
    print(sentence_embeddings_train.shape, train_y.shape)
    print(sentence_embeddings_cv.shape, cv_y.shape)
    print(sentence_embeddings_test.shape, test_y.shape)

    return sentence_embeddings_train, train_y, sentence_embeddings_cv, cv_y, sentence_embeddings_test, test_y
    
def load_bert_embeddings(mode = 'original', classification_type = 'binary'):
    train_X, train_Y = save_embeddings(mode, 'train', classification_type)
    cv_X, cv_Y = save_embeddings(mode, 'dev', classification_type)
    test_X, test_Y = save_embeddings(mode, 'test', classification_type)
    return train_X, train_Y, cv_X, cv_Y, test_X, test_Y

def load_sentiment_dataset(mode = 'original', classification_type = 'binary', emb_type = 'sentence'):
    if(emb_type == 'sentence'):
        return load_sentence_embeddings(mode, classification_type)
    elif(emb_type == 'word'):
        return load_bert_embeddings(mode, classification_type)