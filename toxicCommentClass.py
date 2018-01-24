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

TODO: Remove IP addresses (and possibly user_ids). Maybe all numbers. 
TODO: See the most commonly occuring words when a message is classified as
      different things.
TODO: Add submission of actual test data.
"""

import pandas as pd
import re
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

train = pd.read_csv("data/train.csv")
sampleSub = pd.read_csv("data/sample_submission.csv")
train = train.set_index('id')
sampleSub = sampleSub.set_index('id')
trainX = train[['comment_text']]
trainX.columns = ['msg']
trainY = train.drop('comment_text', axis=1)
classes = list(trainY.columns)


def EDA(trainX, trainY):
    """ All exploratory data analysis will be contained within here """
    classCounts = trainY.sum()
    print("The classes and the number of comments classified as each is:\n" +
          classCounts.to_string())


def addFeatures(data):
    """ Add extra features beyond the bag of n-grams """
    data['pcCaps'] = data['msg'].apply(lambda x:
                                        len(re.findall(r'[A-Z]', x))/len(x))
    return data


def removeURLs(data):
    data = data.replace(r'http\S+', 'website', regex=True).\
                replace(r'www.\S+', 'website', regex=True)
    return data


def removeSpecials(data):
    charDict = {'msg': {'!': ' ', '"': ' ', '\n': ' ',
                                 '@': ''}}
    data = data.replace(charDict, regex=True)
    return data


def cleanData(data):
    data = removeURLs(data)
    data = removeSpecials(data)
    data['msg'] = data['msg'].astype(str)
    return data


trainX = cleanData(trainX)
trainX = addFeatures(trainX)

# Create vectorizer for function to use. Thanks to stackexchange
# https://stackoverflow.com/questions/30653642/combining-bag-of-words-and-other
# -features-in-one-model-using-sklearn-and-pandas

mapper = DataFrameMapper([
    (['pcCaps'], None),
    ('msg', TfidfVectorizer(binary=True, ngram_range=(1, 1),
                                     min_df=0.0005, max_df=0.5))
], sparse=True)
trainXSparse = mapper.fit_transform(trainX)
trainXSparseColumns = mapper.features[0][0] +\
                      mapper.features[1][1].get_feature_names()

# Load the submission data
rTest = pd.read_csv("data/test.csv")
rTest = rTest.set_index('id')
rTest.columns = ['msg']
rTest = cleanData(rTest)
rTest = addFeatures(rTest)
rTestSparse = mapper.transform(rTest)

sTrainX, sTestX, sTrainY, sTestY = train_test_split(
        trainXSparse, trainY, test_size=0.25, random_state=42)

# fit from
# https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb


def prob(x, y_i, y):
    p = x[y == y_i].sum(0)
    return (p + 1) / ((y == y_i).sum() + 1)


def get_mdl(x, y):
    y = y.values
    r = np.log(prob(x, 1, y) / prob(x, 0, y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

score = []

for column in classes:
    m, r = get_mdl(sTrainX, sTrainY[column])
    prediction = pd.DataFrame(m.predict_proba(sTestX.multiply(r))[:,1])
    score.append(log_loss(sTestY[column], prediction))
    m, r = get_mdl(trainXSparse, trainY[column])
    sampleSub[column] = m.predict_proba(rTestSparse.multiply(r))[:,1]

print(score)
print(np.mean(score))

sampleSub.to_csv("submissions/NB2.csv")
