#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: El diablo Dany, Wheelchair Mark, Kim Sam

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
import codecs

vocabulary = codecs.open("brown_vocab_100.txt")
# calculate vocab size
vocab_size = 0
for vocab in vocabulary:
    vocab_size += 1

path = "brown_100.txt"


def calc_tri_proba(w1, w2, w3):
    """
    calc tri gram proba of p(w1 / w2 , w3)
    :param w1: str
    :param w2: str
    :param w3: str
    :return: p(w1 / w2 , w3)
    """
    f = codecs.open(path)
    bigram_count = 0
    trigram_count = 0
    for sentence in f:
        # preprocess + tokenize
        sentence = sentence.lower().rstrip().split()

        for word_index in range(len(sentence) - 1):
            # count bigram
            if w1 == sentence[word_index] and w2 == sentence[word_index + 1]:
                bigram_count += 1
            # count trigram
            if word_index < len(sentence) - 2:
                if w1 == sentence[word_index] and w2 == sentence[word_index + 1] and w3 == sentence[word_index + 2]:
                    trigram_count += 1
    f.close()
    if bigram_count != 0:
        return trigram_count / bigram_count
    else:
        return 0


def calc_tri_proba_with_smoothing(w1, w2, w3, alpha=0.1):
    """
    calc tri gram proba of p(w3 / w1 , w2)
    :param alpha: smoothing parameter
    :param w1: str
    :param w2: str
    :param w3: str
    :return: p(w3 / w1 , w2)
    """
    f = codecs.open(path)
    bigram_count = 0
    trigram_count = 0
    for sentence in f:
        # preprocess + tokenize
        sentence = sentence.lower().rstrip().split()

        for word_index in range(len(sentence) - 1):
            # count bigram
            if w1 == sentence[word_index] and w2 == sentence[word_index + 1]:
                bigram_count += 1
            # count trigram
            if word_index < len(sentence) - 2:
                if w1 == sentence[word_index] and w2 == sentence[word_index + 1] and w3 == sentence[word_index + 2]:
                    trigram_count += 1
    f.close()
    if bigram_count != 0:
        return (trigram_count + alpha) / (bigram_count + alpha * vocab_size)
    else:
        return 0


calc_for_this = [
    ("in", "the", "past"),
    ("in", "the", "time"),
    ("the", "jury", "said"),
    ("the", "jury", "recommended"),
    ("jury", "said", "that"),
    ("agriculture", "teacher", ",")
]

# TEST STUFF
# print(f'{calc_tri_proba("in", "the", "past")}')
# print(f'{calc_tri_proba_with_smoothing("in", "the", "past")}')

# DONE: writeout bigram probabilities
with open("trigram_probs.txt", "w") as file:
    for tri1, tri2, tri3 in calc_for_this:
        result = calc_tri_proba(tri1, tri2, tri3)
        result_smooth = calc_tri_proba_with_smoothing(tri1, tri2, tri3)
        print(f"{tri1} {tri2} {tri3}: {result} {result_smooth}")
        file.write(f"{result} {result_smooth}\n")
