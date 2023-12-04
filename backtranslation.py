#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:25:14 2023

@author: isabelbeaulieu
"""

'''
In this code, sentences were translated from English, to Korean, back to English.
A logistic regression model was then built after cleaning and vectorizing the back-translated
sentences. 
The translator could not handle the entire dataset, so 100 training sentences and 20 testing were used.
The accuracy was bad for both but the confusion matricies the same, which shows that backtranslation
produces the exact same results.
To really know for sure, this should be done on the entire dataset
'''

#import packages
import os
import pandas as pd
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


path= '/Users/isabelbeaulieu/Desktop/Data Mining'
os.chdir(path)

train = pd.read_csv("train.csv")
train['emotion'] = train['emotion'].astype('category')

test = pd.read_csv("test.csv")
test['emotion'] = test['emotion'].astype('category')

val = pd.read_csv("val.csv")
val['emotion'] = val['emotion'].astype('category')

#back translate some training and testing sentences
import googletrans
from BackTranslation import BackTranslation

trans = BackTranslation()

back_trans_train = []
for i in range(0,100):
    sent = train['sentence'][i]
    result = trans.translate(sent, src='en',tmp='ko')
    back_trans_train.append(result.result_text)
    
len(back_trans_train)

train['emotion'][0:100].value_counts()

back_trans_test = []
for i in range(0,20):
    sent = test['sentence'][i]
    result = trans.translate(sent, src='en',tmp='ko')
    back_trans_test.append(result.result_text)
    
test['emotion'][0:20].value_counts()


#cleaning functions
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


#best model found previously was logistic regression removing stopwords and TFIDF
#clean the back translated sentences
train_trans = pd.DataFrame()
train_trans['sentence'] = back_trans_train
train_trans['emotion'] = train['emotion'][0:100]

test_trans = pd.DataFrame()
test_trans['sentence'] = back_trans_test
test_trans['emotion'] = test['emotion'][0:20]

#without stopwords
train_trans['clean_text'] = train_trans['sentence'].apply(lambda x: finalpreprocess_stop(x))
test_trans['clean_text'] = test_trans['sentence'].apply(lambda x: finalpreprocess_stop(x))
X_train_trans, y_train_trans = train_trans.clean_text, train_trans.emotion
X_test_trans, y_test_trans = test_trans.clean_text, test_trans.emotion

tfidf_vectorizer = TfidfVectorizer(use_idf=True)

X_train_vectors_tfidf_trans = tfidf_vectorizer.fit_transform(X_train_trans) 
X_test_vectors_tfidf_trans = tfidf_vectorizer.transform(X_test_trans)

model = LogisticRegression(C=10, solver='liblinear')
model.fit(X_train_vectors_tfidf_trans,y_train_trans)
pred = model.predict(X_test_vectors_tfidf_trans)


sns.set() # use seaborn plotting style
mat = confusion_matrix(y_test_trans, pred)
sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=y_train_trans.unique(),yticklabels=y_train_trans.unique())
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()
accuracy = accuracy_score(y_test_trans, pred)
print("Accuracy: {}".format(accuracy)) #Accuracy is 35%

#compare this to normal data

training = pd.DataFrame()
training['sentence'] = train['sentence'][0:100]
training['emotion'] = train['emotion'][0:100]

testing = pd.DataFrame()
testing['sentence'] = test['sentence'][0:20]
testing['emotion'] = test['emotion'][0:20]

training['clean_text'] = training['sentence'].apply(lambda x: finalpreprocess_stop(x))
testing['clean_text'] = testing['sentence'].apply(lambda x: finalpreprocess_stop(x))
X_train, y_train = training.clean_text, training.emotion
X_test, y_test = testing.clean_text, testing.emotion

tfidf_vectorizer = TfidfVectorizer(use_idf=True)

X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train) 
X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)

model = LogisticRegression(C=10, solver='liblinear')
model.fit(X_train_vectors_tfidf,y_train)
pred_normal = model.predict(X_test_vectors_tfidf)

sns.set() # use seaborn plotting style
mat = confusion_matrix(y_test, pred_normal)
sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=y_train.unique(),yticklabels=y_train.unique())
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()
accuracy = accuracy_score(y_test, pred_normal)
print("Accuracy: {}".format(accuracy)) #Accuracy is 25%
