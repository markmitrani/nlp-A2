#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: El diablo Dany, Wheelchair Mark, Kim Sam

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
import codecs

vocab = codecs.open("brown_vocab_100.txt")

# load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    # DONE: import part 1 code to build dictionary
    word_index_dict.setdefault(line.rstrip(), i)

f = codecs.open("brown_100.txt")

# counts[current_word, previous_word]
counts = np.zeros((len(word_index_dict), len(word_index_dict)))  # DONE: initialize numpy 0s array

# DONE: iterate through file and update counts
for sentence in f:
    previous_word = '<s>'
    for current_word in sentence.rstrip().lower().split()[1:]:  # [1:] to skip '<s>'
        current_word_index = word_index_dict[current_word]
        previous_word_index = word_index_dict[previous_word]
        counts[previous_word_index][current_word_index] += 1
        previous_word = current_word

# New in problem4: 0.1-smoothing
counts += 0.1

# DONE: normalize counts
probs = normalize(counts, norm='l1', axis=1)

# DONE: writeout bigram probabilities
with open("smooth_probs.txt", "w") as file:
    file.write(f'{probs[word_index_dict["all"]][word_index_dict["the"]]}\n')
    file.write(f'{probs[word_index_dict["the"]][word_index_dict["jury"]]}\n')
    file.write(f'{probs[word_index_dict["the"]][word_index_dict["campaign"]]}\n')
    file.write(f'{probs[word_index_dict["anonymous"]][word_index_dict["calls"]]}\n')

f.close()