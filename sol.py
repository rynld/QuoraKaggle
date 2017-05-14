import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
#now = datetime.datetime.now()

train = pd.read_csv('input/train.csv')#.head(100)
test = pd.read_csv('input/test.csv')#.head(100)


def one_hot_encoder(q1_train, q2_train, q1_test, q2_test):

    lb = LabelBinarizer()
    lb.fit(list(q1_train) + list(q2_train) + list(q1_test) + list(q2_test))

    return lb.transform(q1_train), lb.transform(q2_train), lb.transform(q1_test), lb.transform(q2_test)

q1_trn, q2_trn, q1_tst, q2_tst = one_hot_encoder(train["question1"].values, train["question2"].values, test["question1"], test["question2"])

print("Transformation complete")

voc = np.shape(q1_trn)[1]

X = pd.concat((pd.DataFrame(q1_trn, columns=["q1_" + str(i) for i in range(voc)]), pd.DataFrame(q2_trn, columns=["q2_" + str(i) for i in range(voc)])),axis=1)
Y = train["is_duplicate"]

X_test = pd.concat((pd.DataFrame(q1_tst, columns=["q1_" + str(i) for i in range(voc)]), pd.DataFrame(q2_tst, columns=["q2_" + str(i) for i in range(voc)])),axis=1)

xgb_params = {#

    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'silent': 1
}

dtrain = xgb.DMatrix(X, Y)
dtest = xgb.DMatrix(X_test)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50)

#cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)

output = pd.DataFrame({'id': test["test_id"], 'price_doc': y_predict})

current_date = datetime.datetime.now()
output.to_csv('output/xgbSub{0}-{1}-{2}-{3}.csv'.format(current_date.day,current_date.hour,current_date.minute,current_date.second), index=False)