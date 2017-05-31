import numpy as np
import pandas as pd
import xgbst as xgb
import datetime

tr = pd.read_csv('./input/train.csv')#.head(100)
te = pd.read_csv('./input/test.csv')#.head(100)
from nltk.corpus import stopwords
SCALE = 0.3627


def word_match_share(x):
    '''
    The much-loved word_match_share feature.

    Args:
        x: source data with question1/2

    Returns:
        word_match_share as a pandas Series
    '''
    stops = set(stopwords.words('english'))
    q1 = x.question1.fillna(' ').str.lower().str.split()
    q2 = x.question2.fillna(' ').str.lower().str.split()
    q1 = q1.map(lambda l : set(l) - stops)
    q2 = q2.map(lambda l : set(l) - stops)
    q = pd.DataFrame({'q1':q1, 'q2':q2})
    q['len_inter'] = q.apply(lambda row : len(row['q1'] & row['q2']), axis=1)
    q['len_tot'] = q.q1.map(len) + q.q2.map(len)
    return (2 * q.len_inter / q.len_tot).fillna(0)


def bin_model(tr, te, bins=100, vpos=1, vss=3):
    '''
    Runs a Pandas table model using the word_match_share feature.

    Args:
        tr: pandas DataFrame with question1/2 in it
        te: test data frame
        bins: word shares are rounded to whole numbers after multiplying by bins.
        v_pos: number of virtual positives for smoothing (can be non-integer)
        vss: virtual sample size for smoothing (can be non-integer)

    Returns:
        submission in a Pandas Data Frame.
    '''
    tr['word_share'] = word_match_share(tr)
    tr['binned_share'] = (bins * tr.word_share).round()
    pos = tr.groupby('binned_share').is_duplicate.sum()
    cts = tr.binned_share.value_counts()
    te['word_share'] = word_match_share(te)
    te['binned_share'] = (bins * te.word_share).round()
    te_pos = te.binned_share.map(pos, na_action='ignore').fillna(0)
    te_cts = te.binned_share.map(cts, na_action='ignore').fillna(0)

    prob = (te_pos + vpos) / (te_cts + vss)
    odds = prob / (1 - prob)
    scaled_odds = SCALE * odds
    scaled_prob = scaled_odds / (1 + scaled_odds)
    sub = te[['test_id']].copy()
    sub['is_duplicate'] = scaled_prob
    return sub

def training(train, test):

    test_id = test["test_id"]
    train.drop(["question1", "question2", "id", "qid1", "qid2"], axis=1, inplace=True)
    test.drop(["question1", "question2", "test_id"], axis=1, inplace=True)

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

    # cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

    num_boost_rounds = len(cv_output)
    model = xgb.train(dict(xgb_params), dtrain, num_boost_round=num_boost_rounds)

    y_predict = model.predict(dtest)

    output = pd.DataFrame({'test_id': test_id, 'is_duplicate': y_predict})

    current_date = datetime.datetime.now()
    output.to_csv('output/xgbSub{0}-{1}-{2}-{3}.csv'.format(current_date.day, current_date.hour, current_date.minute,
                                                            current_date.second), index=False)

sub = bin_model(tr, te)
training(tr,te)

#sub.to_csv('no_ml_model.csv', index=False, float_format='%.6f')