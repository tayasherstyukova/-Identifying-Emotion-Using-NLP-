"""
Created on Thu Nov 29 

@author:  isabelbeaulieu and tayasherstiukova
"""

import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
import tensorflow as tf


train = pd.read_csv("train.csv")
train['emotion'] = train['emotion'].astype('category')
sentences_train = train['sentence']

test = pd.read_csv('test.csv')
sentences_test = test['sentence']

data_classes = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
train['label'] = train['emotion'].apply(data_classes.index)
y_train = train['label']

test['label'] = test['emotion'].apply(data_classes.index)
y_test = test['label']

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 50

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))  
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)


loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

#Training Accuracy: 0.9973
#Testing Accuracy:  0.8020