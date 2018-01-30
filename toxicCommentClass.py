# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:16:32 2018

@author: George L. Roberts

Use manually classified data from Wikipedia's talk page edits to classify
conversations according to different 'toxicity' features.
This is an example of multi-label classification because each message can
belong to multiple classes.

TODO: EDA
        * See the most commonly occuring  two word sequences for each class.

TODO: Add features.
        * Number of special characters.
        * Number of times a full-stop is followed by a lowercase letter?

TODO: Decide what to do with numbers.
TODO: Remove double spaces
TODO: Save cleaned data so it doesn't need to be loaded every time the fit is
        run
TODO: Tune hyperparameters: What works well on the leaderboard isn't
        necessarily what words well in the model due to their method of
        separation of test and training data.
TODO: Truncated SVD before fitting.
"""
# %% Load packages and data

import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import itertools
from scipy import sparse, io
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter


plt.rcParams.update({'font.size': 22})
stops = set(stopwords.words("english"))

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

# %% Clean data and EDA

classes = list(trainY.columns)


def addFeatures(data):
    """ Add extra features beyond the bag of n-grams """
    data = data.assign(pcCaps=data['msg'].
                       apply(lambda x: len(re.findall(r'[A-Z]', x)) / len(x)))
    data = data.assign(noExclam=data['msg'].
                       apply(lambda x: len(re.findall(r'!', x))))
    data = data.assign(noQues=data['msg'].
                       apply(lambda x: len(re.findall(r'\?', x))))
    data = data.assign(msgLength=data['msg'].
                       apply(lambda x: len(x)))
    data = data.assign(noWords=data['msg'].
                       apply(lambda x: len(x.split())))
    data = data.assign(noUniqueWords=data['msg'].
                       apply(lambda x: len(set(x.split()))))

    def avgWordLengthFn(x):
        wordList = [len(word) for word in x.split()]
        if not wordList:
            return 0
        else:
            return np.mean(wordList)
    data = data.assign(avgWordLength=data['msg'].apply(avgWordLengthFn))

    def pcWordCapsFn(x):
        wordList = [word.isupper() for word in x.split()]
        if not wordList:
            return 0
        else:
            return np.mean(wordList)
    data = data.assign(pcWordCaps=data['msg'].apply(pcWordCapsFn))

    def pcWordTitleFn(x):
        wordList = [word.istitle() for word in x.split()]
        if not wordList:
            return 0
        else:
            return np.mean(wordList)
    data = data.assign(pcWordTitle=data['msg'].apply(pcWordTitleFn))

    return data


def EDA(trainX, trainY, classes):
    """ All exploratory data analysis will be contained within here """
    trainX = addFeatures(trainX)

    # First look at the classes and their occurrences
    classCounts = trainY.sum(axis=0)
    # Add total messages and total clean messages
    classCounts = classCounts.append(pd.Series(len(trainY),
                                               index=['Total_messages']))
    classCounts = classCounts.append(pd.Series(
            len(trainY[trainY.sum(axis=1) == 0]), index=['Clean_messages']))
    newClasses = list(classes)
    newClasses.extend(['Total_messages', 'Clean_messages'])

    fig, ax = plt.subplots(figsize=(25, 20))
    sns.barplot(newClasses, classCounts)

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

    a = trainX.drop('msg', axis=1).join(trainY)
    corr = a.corr()
    fig, ax = plt.subplots(figsize=(15, 15))
    plot = ax.matshow(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True))
    fig.colorbar(plot)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)

    # Find the most common words for each class

    for vulgar in classes:
        temp = trainX[trainY[vulgar] == 1]
        temp = cleanData(pd.DataFrame(temp))['msg']
        temp = temp.apply(lambda x: x.lower())
        temp = temp.apply(lambda x: word_tokenize(x))
        allWords = list(itertools.chain.from_iterable(list(temp.values)))
        filteredWords = [word for word in allWords if word not in stops]
        commonWords = [word for word, word_count in
                       Counter(filteredWords).most_common(10)]
        print("The top 10 most common words in the class " + vulgar + " are: "
              + (', '.join(commonWords)))

    return corr


def cleanData(data):
    """ Remove URLs, special characters and IP addresses """
    data = data.replace(r'http\S+', 'website', regex=True).\
        replace(r'www.\S+', 'website', regex=True)

    # charDict = {'msg': {'!': ' ', '"': ' ', '\n': ' ', '@': ' '}}
    # data = data.replace(charDict, regex=True)

    data = data.replace(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', 'IP',
                        regex=True)

    data = data.replace(r'[^a-zA-Z ]', ' ', regex=True)

    data['msg'] = data['msg'].astype(str)
    return data


def cleanAndSave(trainX, rTest):
    trainX = addFeatures(trainX)
    rTest = addFeatures(rTest)
    trainX = cleanData(trainX)
    rTest = cleanData(rTest)

    trainX.to_csv('data/cleanTrainX.csv')
    rTest.to_csv('data/cleanrTest.csv')

    return trainX, rTest


# corr = EDA(trainX, trainY, classes)
trainX, rTest = cleanAndSave(trainX, rTest)

# %% Fitting

# fit from
# https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=True, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual,
                                       n_jobs=self.n_jobs).fit(x_nb, y)
        return self


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def fitData(classes, trainY):
    """ Transform the data into a sparse matrix and fit it """
    trainX = pd.read_csv('data/cleanTrainX.csv')
    rTest = pd.read_csv('data/cleanrTest.csv')

    def runAndSaveMapper(trainX, rTest):
        mapper = DataFrameMapper([
                (['pcWordCaps', 'pcWordTitle'], None),
                ('msg', TfidfVectorizer(binary=True, ngram_range=(1, 2),
                                        tokenizer=LemmaTokenizer(),
                                        analyzer='word',
                                        stop_words='english',
                                        min_df=3, max_features=None,
                                        use_idf=1, smooth_idf=1,
                                        sublinear_tf=1))
        ], sparse=True)
        trainXSparse = mapper.fit_transform(trainX)
        # trainXSparseColumns = mapper.transformed_names_
        rTestSparse = mapper.transform(rTest)
        io.mmwrite("trainXSparse.mtx", trainXSparse)
        io.mmwrite("rTestSparse.mtx", rTestSparse)

    runAndSaveMapper(trainX, rTest)
    trainXSparse = sparse.csr_matrix(io.mmread("trainXSparse.mtx"))
    rTestSparse = sparse.csr_matrix(io.mmread("rTestSparse.mtx"))

    sTrainX, sTestX, sTrainY, sTestY = train_test_split(
            trainXSparse, trainY, test_size=0.5)

    score = []
    parameters = {'C': np.logspace(-3, 1, 9)}

    for column in classes:
        gsCV = GridSearchCV(NbSvmClassifier(), parameters)
        gsCV.fit(sTrainX, sTrainY[column])
        print(gsCV.best_params_['C'])
        prediction = pd.DataFrame(gsCV.predict_proba(sTestX)[:, 1])
        score.append(log_loss(sTestY[column], prediction))
        model = NbSvmClassifier(C=gsCV.best_params_['C']).fit(trainXSparse,
                                                              trainY[column])
        sampleSub[column] = model.predict_proba(rTestSparse)[:, 1]

    print(score)
    print(np.mean(score))

    sampleSub.to_csv("submissions/NB3.csv")

    return score, sampleSub, trainXSparse


score, sampleSub, trainXSparse = fitData(classes, trainY)
