#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:06:13 2023

@author: isabelbeaulieu
"""

#website for nltk
#https://www.datacamp.com/tutorial/text-analytics-beginners-nltk

import pandas as pd


import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')



from nltk.sentiment.vader import SentimentIntensityAnalyzer


from nltk.stem import WordNetLemmatizer

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



#with stopwords
train['clean_text'] = train['sentence'].apply(lambda x: finalpreprocess(x))
test['clean_text'] = test['sentence'].apply(lambda x: finalpreprocess(x))
val['clean_text'] = val['sentence'].apply(lambda x: finalpreprocess(x))

X_train, y_train = train.clean_text, train.emotion
X_test, y_test = test.clean_text, test.emotion
X_val, y_val = val.clean_text, val.emotion

#nltk.download('all')

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):

    scores = analyzer.polarity_scores(text)

    sentiment = 1 if scores['pos'] > 0 else 0

    return sentiment

train['sentiment'] = train['clean_text'].apply(get_sentiment)

train

df_gb = train.groupby(['emotion', 'sentiment']).size().unstack()

df_gb.plot(kind = 'bar')

#without stopwords
train['clean_text'] = train['sentence'].apply(lambda x: finalpreprocess_stop(x))
test['clean_text'] = test['sentence'].apply(lambda x: finalpreprocess_stop(x))
val['clean_text'] = val['sentence'].apply(lambda x: finalpreprocess_stop(x))

X_train, y_train = train.clean_text, train.emotion
X_test, y_test = test.clean_text, test.emotion
X_val, y_val = val.clean_text, val.emotion

#nltk.download('all')

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):

    scores = analyzer.polarity_scores(text)

    sentiment = 1 if scores['pos'] > 0 else 0

    return sentiment

train['sentiment'] = train['clean_text'].apply(get_sentiment)

train

df_gb = train.groupby(['emotion', 'sentiment']).size().unstack()

df_gb.plot(kind = 'bar')

