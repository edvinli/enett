#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:32:39 2019

@author: edvinli
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.manifold import TSNE
from keras.utils import np_utils
from sklearn.metrics import classification_report
import pickle

data = pd.read_csv('saldo20v03.txt', sep=" ", delimiter="\t",header=None)
data_nouns = data[data[5] == 'nn']
#%%
wordtype = []

for i in range(len(data_nouns)):
    current_word = data_nouns.iloc[i,6]
    if(current_word[4] == 'n'):
        w = 0
    elif(current_word[4] == 'u'):
        w = 1
    else:
        w = 2
        
    wordtype.append(w)
#%%
data_nouns['wordtype'] = wordtype
data_nouns = data_nouns[data_nouns['wordtype'] != 2]
X = data_nouns[4]
y = data_nouns['wordtype']
np.random.seed(9001)
train, validate, test = np.split(data_nouns.sample(frac=1), [int(.6*len(data_nouns)), int(.8*len(data_nouns))]) #0.6 train, 0.2 validation, 0.2 test
X_train = train[4]
y_train = train['wordtype']

X_validate = validate[4]
y_validate = validate['wordtype']

X_test = test[4]
y_test = test['wordtype']

X_test_new = X_test[~X_test.str.endswith(("ing","tion","het","ist","eri","ande","are"))]
y_test_new = y_test[~X_test.str.endswith(("ing","tion","het","ist","eri","ande","are"))]
#%%
tokenizer = Tokenizer(char_level=True,num_words=len(X_train))
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_validate = tokenizer.texts_to_sequences(X_validate)
X_test = tokenizer.texts_to_sequences(X_test)

X_test_new = tokenizer.texts_to_sequences(X_test_new)
#%%
X_train = pad_sequences(X_train, padding='post', maxlen=20)
X_validate = pad_sequences(X_validate, padding='post', maxlen=20)
X_test = pad_sequences(X_test, padding='post', maxlen=20)
X_test_new = pad_sequences(X_test_new, padding='post', maxlen=20)


#%% TSNE

def create_truncated_model(trained_model,vocab_size,embedding_dim):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=20))
    model.add(layers.Bidirectional(layers.LSTM(64,activation='tanh', return_sequences=True)))
    model.add(layers.Flatten())
    for i, layer in enumerate(model.layers):
        layer.set_weights(trained_model.layers[i].get_weights())
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

def calc_tsne(X_test,y_test,model):
    truncated_model = create_truncated_model(model,vocab_size,embedding_dim)
    hidden_features = truncated_model.predict(X_test)
    print("entering t-sne")
    tsne = TSNE(n_components=2, verbose = 1)
    tsne_results = tsne.fit_transform(hidden_features)
    return tsne_results

def plot_tsne(tsne_results):
    y_test_cat = np_utils.to_categorical(y_test, num_classes = 2)
    color_map = np.argmax(y_test_cat, axis=1)
    plt.figure(figsize=(10,10))
    
    cl1 = 1
    indices = np.where(color_map==cl1)
    indices = indices[0]
    plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl1,alpha=0.7)
    
    cl0 = 0
    indices = np.where(color_map==cl0)
    indices = indices[0]
    plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl0,alpha=0.3)

    plt.legend()
    #plt.savefig('tsne_lstm64.png')
    
def calc_metrics(model,X_test,y_test):
    y_preds = model.predict(X_test)
    y_preds = [1 if x>0.5 else 0 for x in y_preds]
    print(classification_report(y_preds, y_test))
#%%
vocab_size = len(wordtype)
embedding_dim = 10
trained_model = load_model("best_model.h5")
tsne_results = calc_tsne(X_test,y_test,trained_model)
#pickle.dump(tsne_results,open("tsne_conv.p","wb"))
print("finished")