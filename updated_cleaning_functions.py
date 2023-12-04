#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:16:59 2023

@author: isabelbeaulieu
"""

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')

from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import os

path= '/Users/isabelbeaulieu/Desktop/Data Mining'
os.chdir(path)

train = pd.read_csv("train.csv")
train['emotion'] = train['emotion'].astype('category')

test = pd.read_csv("test.csv")
test['emotion'] = test['emotion'].astype('category')

val = pd.read_csv("val.csv")
val['emotion'] = val['emotion'].astype('category')

def preprocess(text):
    text = text.lower() 
    text=text.strip()  #get rid of leading/trailing whitespace 
    text=re.compile('<.*?>').sub('', text) #Remove HTML tags/markups
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) #[0-9] matches any digit (0 to 10000...)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) #matches any digit from 0 to 100000..., \D matches non-digits
    text = re.sub(r'\s+',' ',text) #\s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace 
    
    return text

def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

wl = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def finalpreprocess_stop(string):
    return lemmatizer(stopword(preprocess(string)))

def finalpreprocess(string):
    return lemmatizer(preprocess(string))

'''Call finalpreprocess() if you want to keep stopwords
Call finalpreprocess_stop() to remove stopwords'''

train['clean_text'] = train['sentence'].apply(lambda x: finalpreprocess(x))
test['clean_text'] = test['sentence'].apply(lambda x: finalpreprocess(x))
val['clean_text'] = val['sentence'].apply(lambda x: finalpreprocess(x))

X_train, y_train = train.clean_text, train.emotion
X_test, y_test = test.clean_text, test.emotion
X_val, y_val = val.clean_text, val.emotion

'''Vectorizers'''
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
count_vectorizer = CountVectorizer()

'''Call for tfidf vs countvect'''
#TFIDF
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train) #tfidf runs on non-tokenized sentences 
# Only transform x_test (not fit and transform)
X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test) #Don't fit() your TfidfVectorizer to your test data: it will 
#change the word-indexes & weights to match test data. Rather, fit on the training data, then use the same train-data-
#fit model on the test data, to reflect the fact you're analyzing the test data only based on what was learned without 
#it, and the have compatible
X_val_vectors_tfidf = tfidf_vectorizer.transform(X_val)

#CountVect
X_train_vectors_cv = count_vectorizer.fit_transform(X_train)
X_test_vectors_cv = count_vectorizer.transform(X_test)
X_val_vectors_cv = count_vectorizer.transform(X_val)

#Example with SVC and tfidf
from sklearn.svm import SVC
model = SVC()
model.fit(X_train_vectors_tfidf, y_train)
predicted_categories = model.predict(X_test_vectors_tfidf)
sns.set() # use seaborn plotting style
mat = confusion_matrix(y_test, predicted_categories)
sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=y_train.unique(),yticklabels=y_train.unique())
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()
accuracy = accuracy_score(y_test, predicted_categories)
print("Accuracy: {}".format(accuracy))

#took less than a min to run

#now try with countvect intstead
model = SVC()
model.fit(X_train_vectors_cv, y_train)
predicted_categories = model.predict(X_test_vectors_cv)
sns.set() # use seaborn plotting style
mat = confusion_matrix(y_test, predicted_categories)
sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=y_train.unique(),yticklabels=y_train.unique())
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()
accuracy = accuracy_score(y_test, predicted_categories)
print("Accuracy: {}".format(accuracy))