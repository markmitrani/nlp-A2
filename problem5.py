#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: El diablo Dany, Wheelchair Mark, Kim Sam

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
import codecs

vocab = codecs.open("brown_vocab_100.txt")

# load the indices dictionary
word_index_dict = {}

f = codecs.open("brown_100.txt")


def calc_tri_proba(a, b, c):
    """
    calc tri gram proba of p(a / b , c)
    :param a: str
    :param b: str
    :param c: str
    :return: p(a / b , c)
    """
    # p(past / in, the)
    # Count(in the past)/ Count(in the)
    # count b, c
    bigram_count = 0
    trigram_count = 0
    for sentence in f:
        # preprocess + tokenize
        sentence = sentence.lower().rstrip().split()
        for word_index in range(len(sentence) - 1):
            # count bigram
            if b == sentence[word_index] and c == sentence[word_index + 1]:
                bigram_count += 1
            # count trigram
            if word_index < len(sentence) - 2:
                if a == sentence[word_index] and b == sentence[word_index + 1] and c == sentence[word_index + 2]:
                    trigram_count += 1
    if bigram_count != 0:
        return trigram_count / bigram_count
    else:
        return 0


print(f'{calc_tri_proba("past", "in", "the")}')
f.close()
quit()

# DONE: writeout bigram probabilities
with open("trigram_probs.txt", "w") as file:
    file.write(f'{probs[word_index_dict["the"]][word_index_dict["in"]][word_index_dict["past"]]}\n')

f.close()
