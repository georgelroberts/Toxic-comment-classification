# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:16:32 2018

@author: George L. Roberts

Use manually classified data from Wikipedia's talk page edits to classify
conversations according to different 'toxicity' features.
This is an example of multi-label classification because each message can
belong to multiple classes.

TODO: Preliminary EDA on data.
TODO: Add a comment length feature.
TODO: Look at https://www.kdnuggets.com/2017/12/general-approach-preprocessing-
      text-data.html for preprocessing text.
TODO: See the most commonly occuring words when a message is classified as
      toxic.
TODO: Run a simple pipeline using TF-IDF and a naive Bayes' classifier. Multi-
      label so do separately for each class.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score

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

trainX, testX, trainY, testY = train_test_split(
        trainX, trainY, test_size=0.33, random_state=42)

textClf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB())
                    ])


for column in classes:
    textClf.fit(trainX['comment_text'], trainY[column])
    print(cross_val_score(textClf, testX['comment_text'], testY[column], 
                          verbose=1).mean())



