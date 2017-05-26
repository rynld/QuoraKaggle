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


def word_match_share(x, y):
    set_a = set(x)
    set_b = set(y)

    cnt1 = sum([1 for a in set_a if a in set_b])
    cnt2 = sum([1 for b in set_b if b in set_a])
    return (cnt1 + cnt2)/(len(x) + len(y) + 0.0)