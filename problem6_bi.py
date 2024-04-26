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

# PROBLEM 3 START
vocab = codecs.open("brown_vocab_100.txt")
word_index_dict = {}
for i, line in enumerate(vocab):
    word_index_dict.setdefault(line.rstrip(), i)
f = codecs.open("brown_100.txt")
counts = np.zeros((len(word_index_dict), len(word_index_dict)))  # DONE: initialize numpy 0s array
for sentence in f:
    previous_word = '<s>'
    for current_word in sentence.rstrip().lower().split()[1:]:  # [1:] to skip '<s>'
        current_word_index = word_index_dict[current_word]
        previous_word_index = word_index_dict[previous_word]
        counts[previous_word_index][current_word_index] += 1
        previous_word = current_word
probs = normalize(counts, norm='l1', axis=1)
f.close()
# PROBLEM 3 END

# PROBLEM 6 for bigram:
bigram = probs
toy_sentences = codecs.open("toy_corpus.txt")
result_txt = open("bigram_eval.txt","w")
for toy_sentence in toy_sentences:
    print(f"for sentence: {toy_sentence}")
    toy_sentence = toy_sentence.lower().rstrip().split()
    sentprob = 1
    sent_len = len(toy_sentence)-1 # bigram count
    previous_word = '<s>'
    for toy_word in toy_sentence[1:]:
        sentprob *= bigram[word_index_dict[previous_word]][word_index_dict[toy_word]]
        previous_word = toy_word
    if sentprob != 0:
        perplexity = 1 / (pow(sentprob, 1.0 / sent_len))
    else:
        perplexity = "Probability is 0, perplexity can't be computed."
    print(f"we get prob: {sentprob}")
    print(f"we get perplexity: {perplexity}")
    result_txt.write(f"{perplexity}\n")

result_txt.close()
