import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, preprocessing
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import datetime


train = pd.read_csv('input/train.csv', dtype={"question1":str,"question2":str}).head(1000)
test = pd.read_csv('input/test.csv').head(100)

train.fillna("", inplace=True)

res = CountVectorizer(stop_words="english").fit_transform(train["question1"].values + train["question2"].values)

pred = []
for train_index, test_index in StratifiedKFold(n_splits=8).split(res, train["is_duplicate"].values):
    lr = LogisticRegression()
    lr.fit(res[train_index], train["is_duplicate"].values[train_index])
    y_pred = lr.predict_proba(res[test_index])

    pred.append(log_loss(train["is_duplicate"].values[test_index], y_pred))


print(np.mean(pred))


print("-"*20)


bag_of_words = CountVectorizer(stop_words="english")
bag_of_words.fit_transform(train["question1"].values + train["question2"].values)

X = -(bag_of_words.transform(train["question1"].values) != bag_of_words.transform(train["question1"].values)).astype(int)

pred = []
for train_index, test_index in StratifiedKFold(n_splits=8).split(X, train["is_duplicate"].values):
    lr = LogisticRegression()
    lr.fit(X[train_index], train["is_duplicate"].values[train_index])
    y_pred = lr.predict_proba(X[test_index])

    pred.append(log_loss(train["is_duplicate"].values[test_index], y_pred))


print(np.mean(pred))
