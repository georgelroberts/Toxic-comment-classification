# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 07:19:22 2018

@author: George L. Roberts

Use Gensim word embeddings to classify wikipedia talk page comments

TODO: Add gridsearchcv to tune hyperparameters
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from collections import defaultdict

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


def cleanData(data):
    """ Remove URLs, special characters and IP addresses """
    data = data.replace(r'http\S+', 'website', regex=True).\
        replace(r'www.\S+', 'website', regex=True)

    # charDict = {'msg': {'!': ' ', '"': ' ', '\n': ' ', '@': ' '}}
    # data = data.replace(charDict, regex=True)

    data = data.replace(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', 'IP',
                        regex=True)

    data = data.replace(r'\n\t', ' ', regex=True)
    data = data.replace(r'\'', '', regex=True)
    data = data.replace(r'[^a-zA-Z ]', ' ', regex=True)
    data['msg'] = data['msg'].str.strip()
    data['msg'] = data['msg'].astype(str)
    data['msg'] = data['msg'].str.lower()

    return data


trainX = cleanData(trainX)

allWords = trainX.msg.apply(lambda x: word_tokenize(x))
stemmer = SnowballStemmer("english")
allWords = allWords.apply(lambda x: [stemmer.stem(y) for y in x])
allWords = allWords.tolist()

model = Word2Vec(allWords, size=300, window=2, min_count=50, workers=4)
vectors = model.wv
print(vectors.most_similar('hitler'))

# Much of the modelling code is from
# http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
w2v = dict(zip(model.wv.index2word, model.wv.syn0))


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


def fitData(classes, trainX, trainY):
    """ Transform the data into a sparse matrix and fit it """

    sTrainX, sTestX, sTrainY, sTestY = train_test_split(
            trainX, trainY, test_size=0.25)

    w2v_tfid = TfidfEmbeddingVectorizer(w2v)
    w2v_tfid.fit(sTrainX['msg'].tolist(), sTrainY['toxic'].tolist())
    sTrainXw2v = w2v_tfid.transform(sTrainX['msg'].tolist())
    sTestXw2v = w2v_tfid.transform(sTestX['msg'].tolist())
    rTestXw2v = w2v_tfid.transform(rTest['msg'].tolist())

    etree_w2v_tfidf = ExtraTreesClassifier(verbose=1)
    parameters = {'n_estimators': [10, 100, 500]}
    score = []

    for column in classes:
        gsCV = GridSearchCV(etree_w2v_tfidf, parameters)
        gsCV.fit(sTrainXw2v, sTrainY[column])
        print(gsCV.best_estimator_)
        prediction = pd.DataFrame(gsCV.predict_proba(sTestXw2v)[:, 1])
        score.append(log_loss(sTestY[column], prediction))

        sampleSub[column] = gsCV.predict_proba(rTestXw2v)[:, 1]

    print(score)
    print(np.mean(score))

    sampleSub.to_csv("submissions/Etreew2V_3.csv")

    return score


score = fitData(classes, trainX, trainY)
