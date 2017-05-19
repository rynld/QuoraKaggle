import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import datetime
from TextFunctionality import *
from Algo import *

train = pd.read_csv('input/train.csv', dtype=str)#.head(100)
test = pd.read_csv('input/test.csv', dtype=str)#.head(100)

test_id = test["test_id"]
test.drop("test_id", axis=1, inplace=True)


def clean_text(df):

    df.fillna("", inplace=True)
    df["question1"] = df["question1"].map(lambda x: text_to_wordlist(x, stem_words=True))
    df["question2"] = df["question2"].map(lambda x: text_to_wordlist(x, stem_words=True))

clean_text(train)
clean_text(test)

tf = TfidfVectorizer(analyzer="word", stop_words="english")
tf.fit(list(train["question1"].values) + list(train["question2"].values) + list(test["question1"].values) + list(test["question2"].values))


def add_features(df):

    df["q1_len"] = df["question1"].map(lambda x: len(str(x).split(" ")))
    df["q2_len"] = df["question2"].map(lambda x: len(str(x).split(" ")))

    df["cosine_sim"] = df.apply(lambda row: cosine_similarity(tf.transform([row["question1"]]).todense(),
                                                              tf.transform([row["question2"]]).todense()), axis = 1)

    df["manhattan_sim"] = df.apply(lambda row: minkosky_distance(tf.transform([row["question1"]]).todense(),
                                                              tf.transform([row["question2"]]).todense(), 1), axis=1)

    df["euclidean_sim"] = df.apply(lambda row: minkosky_distance(tf.transform([row["question1"]]).todense(),
                                                                  tf.transform([row["question2"]]).todense(), 1), axis=1)

    df.drop(["question1", "question2"], axis=1, inplace=True)

    for c in ["qid1", "qid2", "id"]:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)


add_features(train)
add_features(test)


xgb_params = {

    'eta': 0.04,
    'max_depth': 7,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'silent': 1
}

dtrain = xgb.DMatrix(train.drop("is_duplicate", axis=1).values, train["is_duplicate"].astype(float).values)
dtest = xgb.DMatrix(test.values)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50)

#cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)

output = pd.DataFrame({'test_id': test_id, 'is_duplicate': y_predict})

current_date = datetime.datetime.now()
output.to_csv('output/xgbSub{0}-{1}-{2}-{3}.csv'.format(current_date.day,current_date.hour,current_date.minute,current_date.second), index=False)