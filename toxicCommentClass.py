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
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

train = pd.read_csv("data/train.csv")
sampleSub = pd.read_csv("data/sample_submission.csv")
train = train.set_index('id')
sampleSub = sampleSub.set_index('id')
trainX = train[['comment_text']]
trainX.columns = ['msg']
trainY = train.drop('comment_text', axis=1)
rTest = pd.read_csv("data/test.csv")
rTest = rTest.set_index('id')
rTest.columns = ['msg']

classes = list(trainY.columns)


def addFeatures(data):
    """ Add extra features beyond the bag of n-grams """
    data = data.assign(pcCaps = data['msg'].apply(lambda x:
                               len(re.findall(r'[A-Z]', x))/len(x)))
    data = data.assign(noExclam = data['msg'].apply(lambda x:
                                         len(re.findall(r'!', x))))
    data = data.assign(noQues = data['msg'].apply(lambda x:
                                len(re.findall(r'\?', x))))
    return data


trainX = addFeatures(trainX)
rTest = addFeatures(rTest)


def EDA(trainX, trainY, classes):
    """ All exploratory data analysis will be contained within here """

    # First look at the classes and their occurrences

    classCounts = trainY.sum()
    sns.barplot(classes, classCounts)

    # Now lets extract some short random comments from each class

    for vulgar in classes:
        temp = trainX[trainY[vulgar] == 1]['msg']
        temp = temp[temp.str.len() < 100]
        noComments = 2
        randomNos = random.sample(range(1, len(temp)), noComments)
        print("Vulgarity: " + vulgar)
        print(temp.iloc[randomNos].values)

    # Look at how much the extra features I added above affect the
    # classification.

    a = trainX[['pcCaps', 'noExclam', 'noQues']].join(trainY)
    corr = a.corr()
    plt.matshow(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True))
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    return corr


# corr = EDA(trainX, trainY)


def cleanData(data):
    """ Remove URLs, special characters and IP addresses """
    data = data.replace(r'http\S+', 'website', regex=True).\
           replace(r'www.\S+', 'website', regex=True) 

    charDict = {'msg': {'!': ' ', '"': ' ', '\n': ' ', '@': ' '}}
    # data = data.replace(charDict, regex=True)
    
    data = data.replace(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', 'IP',
                        regex=True)
    
    data = data.replace(r'[^a-zA-Z0-9 ]', ' ', regex=True)

    data['msg'] = data['msg'].astype(str)
    return data


trainX = cleanData(trainX)
rTest = cleanData(rTest)

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


def fitData(classes, trainX, trainY, rTest):
    """ Transform the data into a sparse matrix and fit it """
    mapper = DataFrameMapper([
            (['pcCaps'], None),
            ('msg', TfidfVectorizer(binary=True, ngram_range=(1, 1),
                                    min_df=0.0005, max_df=0.5))
    ], sparse=True)
    trainXSparse = mapper.fit_transform(trainX)
    trainXSparseColumns = mapper.features[0][0] +\
                          mapper.features[1][1].get_feature_names()

    rTestSparse = mapper.transform(rTest)

    sTrainX, sTestX, sTrainY, sTestY = train_test_split(
            trainXSparse, trainY, test_size=0.25, random_state=42)

    score = []

    for column in classes:
        m, r = get_mdl(sTrainX, sTrainY[column])
        prediction = pd.DataFrame(m.predict_proba(sTestX.multiply(r))[:, 1])
        score.append(log_loss(sTestY[column], prediction))
        m, r = get_mdl(trainXSparse, trainY[column])
        sampleSub[column] = m.predict_proba(rTestSparse.multiply(r))[:, 1]

    print(score)
    print(np.mean(score))

    sampleSub.to_csv("submissions/NB2.csv")

    return score, sampleSub


score, sampleSub = fitData(classes, trainX, trainY, rTest)
    