#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 08:55:22 2023

@author: isabelbeaulieu
"""

import pandas as pd
import os

path= '/Users/isabelbeaulieu/Desktop/Data Mining'
os.chdir(path)

sentence = []
emotion = []

#training data

sentence = []
emotion = []

with open("train.txt") as f:
    for line in f:
        line = line.split(";")
        sentence.append(line[0])
        emotion.append(line[1].strip('\n'))
        
train = pd.DataFrame()
train['sentence'] = sentence
train['emotion'] = emotion

train


#testing data

sentence = []
emotion = []

with open("test.txt") as f:
    for line in f:
        line = line.split(";")
        sentence.append(line[0])
        emotion.append(line[1].strip('\n'))
        
test = pd.DataFrame()
test['sentence'] = sentence
test['emotion'] = emotion

test

#validation data
sentence = []
emotion = []

with open("val.txt") as f:
    for line in f:
        line = line.split(";")
        sentence.append(line[0])
        emotion.append(line[1].strip('\n'))
        
val = pd.DataFrame()
val['sentence'] = sentence
val['emotion'] = emotion

val

train.to_csv("train.csv", index = False)
test.to_csv("test.csv", index = False)
val.to_csv("val.csv", index = False)


