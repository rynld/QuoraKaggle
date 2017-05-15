import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import datetime
#now = datetime.datetime.now()

train = pd.read_csv('input/train.csv').head(1000)
test = pd.read_csv('input/test.csv').head(10)


tf = TfidfVectorizer(analyzer="word", stop_words="english", ngram_range=(1,2))

tf.fit(train["question1"].values)
print(tf.get_feature_names())
print(tf.transform(["question for brain", "donald trump"]))
exit()
train["q1"] = train["question1"].map(lambda x: len(str(x).split(" ")))
train["q2"] = train["question2"].map(lambda x: len(str(x).split(" ")))

test["q1"] = test["question1"].map(lambda x: len(str(x).split(" ")))
test["q2"] = test["question2"].map(lambda x: len(str(x).split(" ")))



xgb_params = {

    'eta': 0.04,
    'max_depth': 7,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'silent': 1
}

dtrain = xgb.DMatrix(train[["q1","q2"]].values, train["is_duplicate"])
dtest = xgb.DMatrix(test[["q1","q2"]].values)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50)

#cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)

output = pd.DataFrame({'test_id': test["test_id"], 'is_duplicate': y_predict})

current_date = datetime.datetime.now()
output.to_csv('output/xgbSub{0}-{1}-{2}-{3}.csv'.format(current_date.day,current_date.hour,current_date.minute,current_date.second), index=False)