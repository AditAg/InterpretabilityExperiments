import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertPreTrainedModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from transformers import BertConfig
from tqdm.notebook import tqdm
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import re

from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')

max_length_review = 256
batch_size = 8
save_data = False

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

def load_data(mode, split, classification_type):
    file_location = os.path.join(os.getcwd(), '..')
    file_location = os.path.join(file_location, 'counterfactually-augmented-data')
    file_location = os.path.join(file_location, 'sentiment')
    if(mode == 'original'):
        file_location = os.path.join(file_location, 'orig')
        file_name = split + '_labels_imdb.tsv'
    elif(mode == 'counter_factual'):
        file_location = os.path.join(file_location, 'new')
        file_name = 'new_' + split + '.tsv'
    df = pd.read_csv(os.path.join(file_location, file_name), sep = '\t')
    data = df.values
    np.random.shuffle(data)
    data = convert_to_int(data, classification_type)
    print(data.shape)
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

def process_data(data, classification_type):
    reviews = []
    sentences = list(data[:, 2])
    for sen in sentences:
        reviews.append(preprocess_text(sen))
    if(classification_type == 'binary'):
        y = data[:, 0]
    elif(classification_type == 'multi'):
        y = data[:, 1]
        
    print(len(reviews), y.shape)
    return reviews, y
    
class Compressor(nn.Module):
    def __init__(self, max_len):
        super(Compressor, self).__init__()
        self.max_len = max_len
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    
    def forward(self, sent):
        _, pooled_out, hidden_states = self.bert(sent)
        mean_hidden = torch.mean(torch.stack(hidden_states,-1),-1)
        flat_out = mean_hidden.reshape((-1, 768*self.max_len))
        return flat_out

def tokenize_input(text_list, labels):
    out = []
    for i in text_list:
        out.append(tokenizer.encode(i, max_length=256, pad_to_max_length=True))
    return torch.LongTensor(out), torch.LongTensor(labels.astype(np.int64))

def save_embeddings(mode = 'original', split = 'train', classification_type = 'binary'):

    data = load_data(mode, split, classification_type)
    reviews, output = process_data(data, classification_type)
    
    cmp = Compressor(max_length_review)
    cmp = cmp.cuda()
    
    opti = optim.Adam(cmp.parameters(), lr = 2e-5)
    
    tokenized_sentences, labels = tokenize_input(reviews, output)
    print(tokenized_sentences.shape, labels.shape)
    dataset = [(tokenized_sentences[i],labels[i]) for i in range(len(reviews))]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    new_data = np.zeros((1, 768*max_length_review))
    for data in tqdm(dataloader):
        x, y = data
        cmp.zero_grad()
        x = x.cuda()
        outputs = cmp(x)
        print(outputs.shape)
        data2 = outputs.data.cpu().numpy()
        new_data = np.concatenate((new_data, data2), axis = 0)
    
    new_data = new_data[1:]
    print(new_data.shape, output.shape)
    
    if(save_data == True):
        column_names = []
        for i in range(new_data.shape[1]):
            column_names.append("x" + str(i))
        print(len(column_names))
        new_df = pd.DataFrame(new_data, columns = column_names)
        print(new_df.shape)
        
        new_df["Sentiment"] = output
        print(new_df.shape)
        
        if(mode == 'original'):
            loc = 'orig'
        else:
            loc = 'new'
        new_df.to_csv(os.path.join(os.getcwd(), '..', 'counterfactually-augmented-data', 'sentiment', loc, str(split + '_bert.csv')), sep = '\t', index = None)
        
    return new_data, output
    
    
    