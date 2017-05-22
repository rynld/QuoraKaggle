import numpy as np
from scipy import spatial
from sklearn.metrics import jaccard_similarity_score


def cosine_similarity(x, y):
    return 1 - spatial.distance.cosine(x, y)


def jaccard_similarity(x, y):
    return jaccard_similarity_score(x,y)


def minkosky_distance(x, y, p=1):
    return spatial.distance.minkowski(x, y, p)


def common_words(x, y):
    return len(set(x).intersection(set(y)))


def diff_words(x, y):
    return len(x) - common_words(x,y)