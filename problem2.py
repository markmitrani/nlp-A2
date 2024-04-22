#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE

vocab = open("brown_vocab_100.txt")

# load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    word = line.rstrip()
    word_index_dict[word] = i

print(word_index_dict)
f = open("brown_100.txt")

# TODO: initialize counts to a zero vector
counts = np.zeros(len(word_index_dict))

# TODO: iterate through file and update counts
for i, line in enumerate(f):
    words = line.rstrip().split()
    words_lower = [x.lower() for x in words]
    for word in words_lower:
        counts[word_index_dict[word]] += 1

f.close()

# TODO: normalize and writeout counts.
print("Counts before normalization:")
print(counts)

probs = counts / np.sum(counts)
print(probs)
