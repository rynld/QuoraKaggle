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

def calculate_sim(q1, q2):

    cosine = cosine_similarity(q1, q2, dense_output=False).diagonal()
    cosine = np.reshape(cosine, (len(cosine),1))
    euc = euclidean_distances(q1, q2).diagonal()
    euc = np.reshape(euc, (len(euc), 1))
    manh = manhattan_distances(q1, q2).diagonal()
    manh = np.reshape(manh, (len(manh), 1))

    return np.hstack((cosine, euc, manh))



trainDF = pd.read_csv('input/train.csv')#.head(1000)
trainDF = trainDF.fillna("").reset_index(drop=True)

trainDF["question1"] = trainDF["question1"].map(lambda x: text_to_wordlist(x, True, True))
trainDF["question2"] = trainDF["question2"].map(lambda x: text_to_wordlist(x, True, True))

featureExtractionStartTime = time.time()

maxNumFeatures = 300000

# bag of letter sequences (chars)
BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=50, max_features=maxNumFeatures,
                                      analyzer='char', ngram_range=(1,10),
                                      binary=True, lowercase=True)


BagOfWordsExtractor.fit(pd.concat((trainDF.ix[:,'question1'],trainDF.ix[:,'question2'])).unique())

trainQuestion1_BOW_rep = BagOfWordsExtractor.transform(trainDF.ix[:,'question1'])
trainQuestion2_BOW_rep = BagOfWordsExtractor.transform(trainDF.ix[:,'question2'])

#sim = calculate_sim(trainQuestion1_BOW_rep, trainQuestion2_BOW_rep)

lables = np.array(trainDF.ix[:,'is_duplicate'])

featureExtractionDurationInMinutes = (time.time()-featureExtractionStartTime)/60.0
print("feature extraction took %.2f minutes" % (featureExtractionDurationInMinutes))

crossValidationStartTime = time.time()

numCVSplits = 8
numSplitsToBreakAfter = 2

X = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int)
y = lables

logisticRegressor = linear_model.LogisticRegression(C=0.1, solver='sag')

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

    logisticRegressor.fit(X_train_cv, y_train_cv)

    y_train_hat = logisticRegressor.predict_proba(X_train_cv)[:, 1]
    y_valid_hat = logisticRegressor.predict_proba(X_valid_cv)[:, 1]

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

logisticRegressor = linear_model.LogisticRegression(C=0.1, solver='sag',
                                                    class_weight={1: 0.46, 0: 1.32})
logisticRegressor.fit(X, y)

trainingDurationInMinutes = (time.time()-trainingStartTime)/60.0
print('full training took %.2f minutes' % (trainingDurationInMinutes))

testPredictionStartTime = time.time()

testDF = pd.read_csv('input/test.csv')
testDF.ix[testDF['question1'].isnull(),['question1','question2']] = 'random empty question'
testDF.ix[testDF['question2'].isnull(),['question1','question2']] = 'random empty question'

testQuestion1_BOW_rep = BagOfWordsExtractor.transform(testDF.ix[:,'question1'])
testQuestion2_BOW_rep = BagOfWordsExtractor.transform(testDF.ix[:,'question2'])

X_test = -(testQuestion1_BOW_rep != testQuestion2_BOW_rep).astype(int)


# quick fix to avoid memory errors
seperators= [750000,1500000]
testPredictions1 = logisticRegressor.predict_proba(X_test[:seperators[0],:])[:,1]
testPredictions2 = logisticRegressor.predict_proba(X_test[seperators[0]:seperators[1],:])[:,1]
testPredictions3 = logisticRegressor.predict_proba(X_test[seperators[1]:,:])[:,1]
testPredictions = np.hstack((testPredictions1,testPredictions2,testPredictions3))


testPredictionDurationInMinutes = (time.time()-testPredictionStartTime)/60.0
print('predicting on test took %.2f minutes' % (testPredictionDurationInMinutes))


submissionName = 'shallowBenchmark'

submission = pd.DataFrame()
submission['test_id'] = testDF['test_id']
submission['is_duplicate'] = testPredictions
submission.to_csv(submissionName + '.csv', index=False)