import time
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from TextFunctionality import text_to_wordlist
from Algo import *
import scipy
from file_operations import *
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

LOAD = False

trainDF = pd.read_csv('input/train.csv')#.head(1000)
testDF = pd.read_csv('input/test.csv')#.head(1000)

crossValidationStartTime = time.time()
numCVSplits = 8
numSplitsToBreakAfter = 2
maxNumFeatures = 10000

BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=50, max_features=maxNumFeatures,
                                      analyzer='char', ngram_range=(1, 8),
                                      binary=True, lowercase=True)


def calculate_sim(q1, q2):

    cosine = cosine_similarity(q1, q2, dense_output=False).diagonal()
    cosine = np.reshape(cosine, (len(cosine),1))
    euc = euclidean_distances(q1, q2).diagonal()
    euc = np.reshape(euc, (len(euc), 1))
    manh = manhattan_distances(q1, q2).diagonal()
    manh = np.reshape(manh, (len(manh), 1))

    return np.hstack((cosine, euc, manh))


def feature_engineer(df):
    df.fillna('random empty question', inplace=True)
    lastTime = time.time()
    # testDF["question1"] = testDF["question1"].map(lambda x: text_to_wordlist(x, True, True))
    # testDF["question2"] = testDF["question2"].map(lambda x: text_to_wordlist(x, True, True))

    df["q1_len"] = df["question1"].map(lambda x: len(x.split(" ")))
    df["q2_len"] = df["question2"].map(lambda x: len(x.split(" ")))
    df["len_diff"] = df["q1_len"] - df["q2_len"]
    df["len_sum"] = df["q1_len"] + df["q2_len"]
    df["common_words"] = df.apply(
        lambda row: common_words(row["question1"].split(" "), row["question2"].split(" ")), axis=1)
    df["diff_words"] = df.apply(
        lambda row: diff_words(row["question1"].split(" "), row["question2"].split(" ")), axis=1)
    df["word_match_share"] = df.apply(
        lambda row: word_match_share(row["question1"].split(" "), row["question2"].split(" ")), axis=1)
    df["sq_common_words"] = np.power(df["common_words"], 2)
    df["log_common_words"] = np.log1p(df["common_words"])

    print("Create features toke: {0}".format(time.time() - lastTime))
    lastTime = time.time()

    if "is_duplicate" in df.columns:
        BagOfWordsExtractor.fit(pd.concat((df.ix[:, 'question1'], df.ix[:, 'question2'])).unique())

    X_q1 = BagOfWordsExtractor.transform(df.ix[:, 'question1'])
    X_q2 = BagOfWordsExtractor.transform(df.ix[:, 'question2'])

    print("Bag of words toke: {0}".format(time.time() - lastTime))
    lastTime = time.time()

    Y = []
    if 'is_duplicate' in df.columns:
        Y = np.array(df.ix[:, 'is_duplicate'])

    test_id = []
    if 'test_id' in df.columns:
        test_id = np.array(df.ix[:, 'test_id'])

    for c in ["question1", "question2", "qid1", "qid2", "is_duplicate", "id", "test_id"]:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)

    X = -(X_q1 != X_q2).astype(int)

    X = scipy.sparse.hstack((X, df.values)).tocsr()

    if len(Y) != 0:
        return X, Y
    return X, test_id


def get_train():

    if LOAD:
        return load_sparse_csr("xtrain.npz"), load_matrix("ytrain.npy")

    trainDF = pd.read_csv('input/train.csv').head(100)
    trainDF = trainDF.fillna("random empty question")

    # trainDF["question1"] = trainDF["question1"].map(lambda x: text_to_wordlist(x, True, True))
    # trainDF["question2"] = trainDF["question2"].map(lambda x: text_to_wordlist(x, True, True))
    trainDF["q1_len"] = trainDF["question1"].map(lambda x: len(x.split(" ")))
    trainDF["q2_len"] = trainDF["question2"].map(lambda x: len(x.split(" ")))
    trainDF["len_diff"] = trainDF["q1_len"] - trainDF["q2_len"]
    trainDF["len_sum"] = trainDF["q1_len"] + trainDF["q2_len"]
    trainDF["common_words"] = trainDF.apply(lambda row: common_words(row["question1"].split(" "), row["question2"].split(" ")), axis=1)
    trainDF["diff_words"] = trainDF.apply(lambda row: diff_words(row["question1"].split(" "), row["question2"].split(" ")), axis=1)
    trainDF["word_match_share"] = trainDF.apply(lambda row: word_match_share(row["question1"].split(" "), row["question2"].split(" ")), axis=1)
    trainDF["sq_common_words"] = np.power(trainDF["common_words"], 2)
    trainDF["log_common_words"] = np.log1p(trainDF["common_words"])

    featureExtractionStartTime = time.time()

    BagOfWordsExtractor.fit(pd.concat((trainDF.ix[:, 'question1'], trainDF.ix[:, 'question2'])).unique())

    trainQuestion1_BOW_rep = BagOfWordsExtractor.transform(trainDF.ix[:,'question1'])
    trainQuestion2_BOW_rep = BagOfWordsExtractor.transform(trainDF.ix[:,'question2'])

    #sim = calculate_sim(trainQuestion1_BOW_rep, trainQuestion2_BOW_rep)

    lables = np.array(trainDF.ix[:,'is_duplicate'])

    featureExtractionDurationInMinutes = (time.time()-featureExtractionStartTime)/60.0
    print("feature extraction took %.2f minutes" % (featureExtractionDurationInMinutes))

    longitud = len(trainDF["q1_len"].values)
    X = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int)
    X = scipy.sparse.hstack((X, np.reshape(trainDF["q1_len"].values, (longitud,1))))
    X = scipy.sparse.hstack((X, np.reshape(trainDF["q2_len"].values, (longitud, 1))))
    y = lables

    print("Saving training")
    # save_sparse_csr("xtrain", X)
    # save_matrix("ytrain", y)
    return X, y
    

def get_test():

    testDF = pd.read_csv('input/test.csv').head(100)

    if LOAD:
        return load_sparse_csr("xtest.npz"), testDF["test_id"]


    testDF.fillna('random empty question', inplace=True)
    testDF["q1_len"] = testDF["question1"].map(lambda x: len(x.split(" ")))
    testDF["q2_len"] = testDF["question2"].map(lambda x: len(x.split(" ")))

    # testDF["question1"] = testDF["question1"].map(lambda x: text_to_wordlist(x, True, True))
    # testDF["question2"] = testDF["question2"].map(lambda x: text_to_wordlist(x, True, True))

    testQuestion1_BOW_rep = BagOfWordsExtractor.transform(testDF.ix[:, 'question1'])
    testQuestion2_BOW_rep = BagOfWordsExtractor.transform(testDF.ix[:, 'question2'])

    longitud = len(testDF["q1_len"].values)
    X_test = -(testQuestion1_BOW_rep != testQuestion2_BOW_rep).astype(int)
    X_test = scipy.sparse.hstack((X_test, np.reshape(testDF["q1_len"].values, (longitud, 1))))
    X_test = scipy.sparse.hstack((X_test, np.reshape(testDF["q2_len"].values, (longitud, 1))))
    save_sparse_csr("xtest", X_test)

    return X_test, testDF["test_id"]

X, y = feature_engineer(trainDF)
X_test, test_id = feature_engineer(testDF)

#X, y = get_train()

model = linear_model.LogisticRegression(C=0.1,max_iter=1000, solver='sag', class_weight={1: 0.46, 0: 1.32}, verbose=0.8)
#model = SVC(probability=True)
#model = BernoulliNB()
model.fit(X, y)


logRegAccuracy = []
logRegLogLoss = []
logRegAUC = []

print('---------------------------------------------')
stratifiedCV = model_selection.StratifiedKFold(n_splits=numCVSplits, random_state=2)

for k, (trainInds, validInds) in enumerate(stratifiedCV.split(X, y)):
    break
    foldTrainingStartTime = time.time()

    X_train_cv = X[trainInds, :]
    X_valid_cv = X[validInds, :]

    y_train_cv = y[trainInds]
    y_valid_cv = y[validInds]

    model.fit(X_train_cv, y_train_cv)

    y_train_hat = model.predict_proba(X_train_cv)[:, 1]
    y_valid_hat = model.predict_proba(X_valid_cv)[:, 1]

    logRegAccuracy.append(accuracy_score(y_valid_cv, y_valid_hat > 0.5))
    logRegLogLoss.append(log_loss(y_valid_cv, y_valid_hat))
    logRegAUC.append(roc_auc_score(y_valid_cv, y_valid_hat))

    foldTrainingDurationInMinutes = (time.time() - foldTrainingStartTime) / 60.0
    print('fold %d took %.2f minutes: accuracy = %.3f, log loss = %.4f, AUC = %.3f' % (k + 1,
                                                                                       foldTrainingDurationInMinutes,
                                                                                       logRegAccuracy[-1],
                                                                                       logRegLogLoss[-1],
                                                                                       logRegAUC[-1]))
    if k >2:
        break

crossValidationDurationInMinutes = (time.time() - crossValidationStartTime) / 60.0

# print('---------------------------------------------')
# print('cross validation took %.2f minutes' % (crossValidationDurationInMinutes))
# print('mean CV: accuracy = %.3f, log loss = %.4f, AUC = %.3f' % (np.array(logRegAccuracy).mean(),
#                                                                  np.array(logRegLogLoss).mean(),
#                                                                  np.array(logRegAUC).mean()))
# print('---------------------------------------------')

trainingStartTime = time.time()

trainingDurationInMinutes = (time.time()-trainingStartTime)/60.0
print('full training took %.2f minutes' % (trainingDurationInMinutes))

testPredictionStartTime = time.time()

#X_test, test_id = get_test()

# quick fix to avoid memory errors
seperators= [750000,1500000]
#seperators= [75,150]
testPredictions1 = model.predict_proba(X_test[:seperators[0],:])[:,1]
testPredictions2 = model.predict_proba(X_test[seperators[0]:seperators[1],:])[:,1]
testPredictions3 = model.predict_proba(X_test[seperators[1]:,:])[:,1]
testPredictions = np.hstack((testPredictions1,testPredictions2,testPredictions3))


testPredictionDurationInMinutes = (time.time()-testPredictionStartTime)/60.0
print('predicting on test took %.2f minutes' % (testPredictionDurationInMinutes))


submissionName = 'shallowBenchmark'

submission = pd.DataFrame()
submission['test_id'] = test_id
submission['is_duplicate'] = testPredictions
submission.to_csv(submissionName + '.csv', index=False)