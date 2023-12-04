#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:28:21 2023

@author: isabelbeaulieu
"""

import pandas as pd
import re
import nltk
import os
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer




path= '/Users/isabelbeaulieu/Desktop/Data Mining'
os.chdir(path)

train = pd.read_csv("train.csv")
train['dataset'] = 'train'
test = pd.read_csv("test.csv")
test['dataset'] = 'test'
val = pd.read_csv("val.csv")
val['dataset'] = 'val'

df = pd.concat([train, test, val], ignore_index=True, axis=0)


def clean_text_remove_stop(df):
    sentences = []
    for i in range(0,len(df)):
        sent=df["sentence"][i]
        sent=re.sub(r'[,.;@#?!&$\-\']+', ' ', sent, flags=re.IGNORECASE)
        sent=re.sub(' +', ' ', sent, flags=re.IGNORECASE)
        sent=re.sub(r'\"', ' ', sent, flags=re.IGNORECASE)
        sent=re.sub(r'[^a-zA-Z]', " ", sent, flags=re.VERBOSE)
        sent=sent.replace(',', '')
        sent=' '.join(sent.split())
        sent=re.sub("\n|\r", "", sent)
        sent = ' '.join([word for word in sent.split() if word not in stopwords.words("english")])
        sentences.append(sent)
    df['clean'] = sentences
    return df

def clean_text_keep_stop(df):
    sentences = []
    for i in range(0,len(df)):
        sent=df["sentence"][i]
        sent=re.sub(r'[,.;@#?!&$\-\']+', ' ', sent, flags=re.IGNORECASE)
        sent=re.sub(' +', ' ', sent, flags=re.IGNORECASE)
        sent=re.sub(r'\"', ' ', sent, flags=re.IGNORECASE)
        sent=re.sub(r'[^a-zA-Z]', " ", sent, flags=re.VERBOSE)
        sent=sent.replace(',', '')
        sent=' '.join(sent.split())
        sent=re.sub("\n|\r", "", sent)
        sentences.append(sent)
    df['clean'] = sentences
    return df

def CountVect(df):
    sent_list=[]
    for i in range(0,len(df)):
        sent_list.append(df['clean'][i])
        
    MyCountV=CountVectorizer(
        input="content", 
        lowercase=True)
    MyDTM = MyCountV.fit_transform(sent_list)  # create a sparse matrix
    MyDTM = MyDTM.toarray()  # convert to a regular array
    ColumnNames=MyCountV.get_feature_names_out()
    MyDTM_DF=pd.DataFrame(MyDTM,columns=ColumnNames)
    return(MyDTM_DF)

def tfidf(df):
    sent_list=[]
    for i in range(0,len(df)):
        sent_list.append(df['clean'][i])
   
    MyVect_TF=TfidfVectorizer(input='content')
    Vect = MyVect_TF.fit_transform(sent_list)
    
    ColumnNamesTF=MyVect_TF.get_feature_names_out()
    DF_TF=pd.DataFrame(Vect.toarray(),columns=ColumnNamesTF)
     
    return (DF_TF)
    

''' Here is an example of how to use the code above.
Say you want to build a model with the input using tf-idf vectorizer and keeping 
stopwords. After running the code above, this is what you would run.'''

##keeping stopwords - tfidf

clean = clean_text_keep_stop(df)
tf_matrix = tfidf(df)

train_clean = clean[clean['dataset'] == 'train']
train_index = clean[clean['dataset'] == 'train'].index.values.astype(int)
test_clean = clean[clean['dataset'] == 'test']
test_index = clean[clean['dataset'] == 'test'].index.values.astype(int)
val_clean = clean[clean['dataset'] == 'val']
val_index = clean[clean['dataset'] == 'val'].index.values.astype(int)


trainLabel = train_clean['emotion'].astype('category')
testLabel = test_clean['emotion'].astype('category')
valLabel = val_clean['emotion'].astype('category')

train_df = tf_matrix.iloc[train_index]
test_df = tf_matrix.iloc[test_index]
val_df = tf_matrix.iloc[val_index]

from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,classification_report

model = LinearSVC()
model.fit(train_df, trainLabel)

pred = model.predict(test_df)
acc_score = metrics.accuracy_score(pred,testLabel)
prec_score = precision_score(testLabel,pred, average='macro')
recall = recall_score(testLabel, pred,average='macro')
f1 = f1_score(testLabel,pred,average='macro')
matrix = confusion_matrix(testLabel,pred)


print(str('Accuracy: '+'{:04.2f}'.format(acc_score*100))+'%')
print(str('Precision: '+'{:04.2f}'.format(prec_score*100))+'%')
print(str('Recall: '+'{:04.2f}'.format(recall*100))+'%')
print('F1 Score: ',f1)
print('\n')
print(classification_report(testLabel,pred))


###loooking at removing stopwords - tfidf

clean = clean_text_remove_stop(df)
tf_matrix = tfidf(df)

train_clean = clean[clean['dataset'] == 'train']
train_index = clean[clean['dataset'] == 'train'].index.values.astype(int)
test_clean = clean[clean['dataset'] == 'test']
test_index = clean[clean['dataset'] == 'test'].index.values.astype(int)
val_clean = clean[clean['dataset'] == 'val']
val_index = clean[clean['dataset'] == 'val'].index.values.astype(int)


trainLabel = train_clean['emotion'].astype('category')
testLabel = test_clean['emotion'].astype('category')
valLabel = val_clean['emotion'].astype('category')

train_df = tf_matrix.iloc[train_index]
test_df = tf_matrix.iloc[test_index]
val_df = tf_matrix.iloc[val_index]
model = LinearSVC()
model.fit(train_df, trainLabel)

pred = model.predict(test_df)
acc_score = metrics.accuracy_score(pred,testLabel)
prec_score = precision_score(testLabel,pred, average='macro')
recall = recall_score(testLabel, pred,average='macro')
f1 = f1_score(testLabel,pred,average='macro')
matrix = confusion_matrix(testLabel,pred)


print(str('Accuracy: '+'{:04.2f}'.format(acc_score*100))+'%')
print(str('Precision: '+'{:04.2f}'.format(prec_score*100))+'%')
print(str('Recall: '+'{:04.2f}'.format(recall*100))+'%')
print('F1 Score: ',f1)
print('\n')
print(classification_report(testLabel,pred))


###count vect and remove stop
clean = clean_text_remove_stop(df)
cv_matrix = CountVect(df)

train_clean = clean[clean['dataset'] == 'train']
train_index = clean[clean['dataset'] == 'train'].index.values.astype(int)
test_clean = clean[clean['dataset'] == 'test']
test_index = clean[clean['dataset'] == 'test'].index.values.astype(int)
val_clean = clean[clean['dataset'] == 'val']
val_index = clean[clean['dataset'] == 'val'].index.values.astype(int)


trainLabel = train_clean['emotion'].astype('category')
testLabel = test_clean['emotion'].astype('category')
valLabel = val_clean['emotion'].astype('category')

train_df = cv_matrix.iloc[train_index]
test_df = cv_matrix.iloc[test_index]
val_df = cv_matrix.iloc[val_index]
model = LinearSVC()
model.fit(train_df, trainLabel)

pred = model.predict(test_df)
acc_score = metrics.accuracy_score(pred,testLabel)
prec_score = precision_score(testLabel,pred, average='macro')
recall = recall_score(testLabel, pred,average='macro')
f1 = f1_score(testLabel,pred,average='macro')
matrix = confusion_matrix(testLabel,pred)


print(str('Accuracy: '+'{:04.2f}'.format(acc_score*100))+'%')
print(str('Precision: '+'{:04.2f}'.format(prec_score*100))+'%')
print(str('Recall: '+'{:04.2f}'.format(recall*100))+'%')
print('F1 Score: ',f1)
print('\n')
print(classification_report(testLabel,pred))