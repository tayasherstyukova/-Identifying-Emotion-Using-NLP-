
#author: @tayasherstiukova
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

val = pd.read_csv('val.csv')
sentences_val = val['sentence']

test = pd.read_csv('test.csv')
sentences_test = test['sentence']

data_classes = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
train['label'] = train['emotion'].apply(data_classes.index)
y_train = train['label']

val['label'] = val['emotion'].apply(data_classes.index)
y_val = val['label']

test['label'] = test['emotion'].apply(data_classes.index)
y_test = test['label']

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_val = tokenizer.texts_to_sequences(sentences_val)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 50

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_val= pad_sequences(X_val, padding='post', maxlen=maxlen)
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

# model.summary()

# Train the model
history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_val, y_val),
                    batch_size=10)

model.evaluate(X_val, y_val, verbose=1) 

model.evaluate(X_test, y_test, verbose=1) 

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
test_acc = model.evaluate(X_test, y_test, verbose=False)[1]
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.axhline(y=test_acc, color='g', linestyle='--', label='Test accuracy')

plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()