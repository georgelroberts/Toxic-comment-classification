# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:16:32 2018

@author: George L. Roberts

Use manually classified data from Wikipedia's talk page edits to classify
conversations according to different 'toxicity' features.
This is an example of multi-label classification because each message can
belong to multiple classes.

TODO: Preliminary EDA on data.
        * Percentage caps locks
        * Message length
        * Count number of exclamation marks/question marks

TODO: Remove IP addresses (and possibly user_ids)
TODO: See the most commonly occuring words when a message is classified as
      different things.
"""

import pandas as pd
import re
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

train = pd.read_csv("data/train.csv")
realTest = pd.read_csv("data/test.csv")
sampleSub = pd.read_csv("data/sample_submission.csv")
train = train.set_index('id')
trainX = train[['comment_text']]
trainY = train.drop('comment_text', axis=1)


classes = list(trainY.columns)
classCounts = trainY.sum()
print("The classes and the number of comments classified as each is:\n" +
      classCounts.to_string())

trainX['percentCaps'] = trainX['comment_text'].apply(lambda x:
                                           len(re.findall(r'[A-Z]', x))/len(x))


def removeURLs(data):
    data = data.replace(r'http\S+', 'website', regex=True).\
                replace(r'www.\S+', 'website', regex=True)
    return data


def removeSpecials(data):
    charDict = {'comment_text': {'!': ' ', '"': ' ', '\n': ' ',
                                 '@': ''}}
    data = data.replace(charDict, regex=True)
    return data


def cleanData(data):
    data = removeURLs(data)
    data = removeSpecials(data)
    data['comment_text'] = data['comment_text'].astype(str)
    return data


trainX = cleanData(trainX)

# Create vectorizer for function to use. Thanks to stackexchange
# https://stackoverflow.com/questions/30653642/combining-bag-of-words-and-other
# -features-in-one-model-using-sklearn-and-pandas

mapper = DataFrameMapper([
    (['percentCaps'], None),
    ('comment_text', TfidfVectorizer(binary=True, ngram_range=(1, 1),
                                     min_df=0.001))
], sparse=True)
trainXSparse = mapper.fit_transform(trainX)
trainXSparseColumns = mapper.features[0][0] +\
                      mapper.features[1][1].get_feature_names()

trainX, testX, trainY, testY = train_test_split(
        trainXSparse, trainY, test_size=0.25, random_state=42)

textClf = MultinomialNB()
score = []
confusion = 0
for column in classes:
    textClf.fit(trainX, trainY[column])
    prediction = pd.DataFrame(textClf.predict(testX))
    score.append(f1_score(testY[[column]], prediction))
    confusion += confusion_matrix(testY[[column]], prediction)

print(score)
print(np.mean(score))
Confusion = pd.DataFrame(np.reshape(confusion, (1, 4)),
                                    columns=['TrueNeg', 'FalsePos', 'FalseNeg',
                                             'TruePos'])
print(Confusion)
totalPrecision = float(Confusion['TruePos'] / (Confusion['TruePos']
                              + Confusion['FalsePos']))
totalRecall = float(Confusion['TruePos'] / (Confusion['TruePos']
                              + Confusion['FalseNeg']))
totalF1 = float(totalPrecision * totalRecall / (totalPrecision + totalRecall))
print(totalF1)
