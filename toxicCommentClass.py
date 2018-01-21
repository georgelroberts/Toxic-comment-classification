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
TODO: See the most commonly occuring words when a message is classified as
      toxic.
TODO: Run a simple pipeline using TF-IDF and a naive Bayes' classifier. Multi-
      label so do separately for each class.
"""

import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sampleSub = pd.read_csv("data/sample_submission.csv")

class_cols = [name for name in train.columns if name not in ['id',
                                                             'comment_text']]
classCounts = train[class_cols].sum()

print("The classes and the number of comments classified as each is:\n" +
      classCounts.to_string())
