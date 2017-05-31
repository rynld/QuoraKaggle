import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import xgbst as xgb
import datetime
from TextFunctionality import *
from Algo import *

train = pd.read_csv('input/train.csv', dtype=str)#.head(10)
test = pd.read_csv('input/test.csv', dtype=str)#.head(10)

test_id = test["test_id"]
test.drop("test_id", axis=1, inplace=True)


def clean_text(df):

    df.fillna("", inplace=True)
    df["question1"] = df["question1"].map(lambda x: text_to_wordlist(x, stem_words=True))
    df["question2"] = df["question2"].map(lambda x: text_to_wordlist(x, stem_words=True))

print("Cleaning text")

clean_text(train)

tf = TfidfVectorizer(analyzer="word", stop_words="english")
tf.fit(list(train["question1"].values) + list(train["question2"].values))


def add_features(df):

    df["q1_len"] = df["question1"].map(lambda x: len(str(x).split(" ")))
    df["q2_len"] = df["question2"].map(lambda x: len(str(x).split(" ")))

    df["len_diff"] = df["q1_len"] - df["q2_len"]
    df["len_sum"] = df["q1_len"] + df["q2_len"]


    print("Calculating Simularities")
    df["cosine_sim"] = df.apply(lambda row: cosine_similarity(tf.transform([row["question1"]]).todense(),
                                                              tf.transform([row["question2"]]).todense()), axis = 1)

    df["manhattan_sim"] = df.apply(lambda row: minkosky_distance(tf.transform([row["question1"]]).todense(),
                                                              tf.transform([row["question2"]]).todense(), 1), axis=1)

    df["euclidean_sim"] = df.apply(lambda row: minkosky_distance(tf.transform([row["question1"]]).todense(),
                                                                  tf.transform([row["question2"]]).todense(), 2), axis=1)

    df["mink_sim"] = df.apply(lambda row: minkosky_distance(tf.transform([row["question1"]]).todense(),
                                                                 tf.transform([row["question2"]]).todense(), 3), axis=1)

    df["common_words"] = df.apply(lambda row: common_words(row["question1"],row["question2"]), axis=1)

    df["diff_words"] = df.apply(lambda row: diff_words(row["question1"], row["question2"]), axis=1)

    df["sq_common_words"] = np.power(df["common_words"], 2)
    df["log_common_words"] = np.log1p(df["common_words"])

    print("Similarity transformations")
    df["log_cosine"] = np.log1p(df["cosine_sim"])
    df["log_manh"] = np.log1p(df["manhattan_sim"])
    df["log_eucl"] = np.log1p(df["euclidean_sim"])
    df["log_mink"] = np.log1p(df["mink_sim"])

    df["sqrt_cosine"] = np.sqrt(df["cosine_sim"])
    df["sqrt_manh"] = np.sqrt(df["manhattan_sim"])
    df["sqrt_eucl"] = np.sqrt(df["euclidean_sim"])
    df["mink_eucl"] = np.sqrt(df["mink_sim"])

    df.drop(["question1", "question2"], axis=1, inplace=True)

    for c in ["qid1", "qid2", "id"]:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)

print("Adding Features")

add_features(train)
train.to_csv("input/train_mod.csv", index=False)

clean_text(test)
add_features(test)
test.to_csv("input/test_mod.csv", index=False)

xgb_params = {

    'eta': 0.04,
    'max_depth': 7,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'silent': 1
}

print("Training")
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