import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import layers
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

#%%

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
#%%
data = pd.read_csv('saldo20v03.txt', sep=" ", delimiter="\t",header=None)
data_nouns = data[data[5] == 'nn']
#%%

#Find the grammatical gender
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
X = data_nouns[4] #noun
y = data_nouns['wordtype'] #grammatical gender
#%%
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
np.random.seed(9001)
train, validate, test = np.split(data_nouns.sample(frac=1), [int(.6*len(data_nouns)), int(.8*len(data_nouns))]) #0.6 train, 0.2 validation, 0.2 test
X_train = train[4]
y_train = train['wordtype']

X_validate = validate[4]
y_validate = validate['wordtype']

X_test = test[4]
y_test = test['wordtype']

#Create test set without common suffixes
X_test_new = X_test[~X_test.str.endswith(("ing","tion","het","ist","eri"))]
y_test_new = y_test[~X_test.str.endswith(("ing","tion","het","ist","eri"))]

#Tokenize characters
tokenizer = Tokenizer(char_level=True,num_words=len(X_train))
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_validate = tokenizer.texts_to_sequences(X_validate)
X_test = tokenizer.texts_to_sequences(X_test)
X_test_new = tokenizer.texts_to_sequences(X_test_new)

#%% Pad all words to be equal length
X_train = pad_sequences(X_train, padding='post', maxlen=20)
X_validate = pad_sequences(X_validate, padding='post', maxlen=20)
X_test = pad_sequences(X_test, padding='post', maxlen=20)
X_test_new = pad_sequences(X_test_new, padding='post', maxlen=20)
#%% Create a model
vocab_size = len(wordtype)
embedding_dim = 60

model = Sequential()
#model.add(layers.Embedding(input_dim=vocab_size, 
#                           output_dim=embedding_dim, 
#                           input_length=20))
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=20))
model.add(layers.layers.LSTM(64,activation='tanh', return_sequences=True))
model.add(layers.Flatten())
#model.add(layers.Dense(128, activation='tanh'))
#model.add(layers.Dense(64, activation='linear'))
#model.add(layers.Dense(64, activation='linear'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
#%% Train and evaluate model
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None)
mdlchk = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train,
                    epochs=10000,
                    verbose=1,
                    validation_data=(X_validate, y_validate),
                    batch_size=32,
		    callbacks=[es,mdlchk])
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_validate, y_validate, verbose=False)
print("Validation Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test_new, y_test_new, verbose=False)
print("Testing w/ removed Accuracy:  {:.4f}".format(accuracy))


