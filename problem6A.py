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

# DONE: initialize counts to a zero vector
counts = np.zeros(len(word_index_dict))

# DONE: iterate through file and update counts
for i, line in enumerate(f):
    words = line.rstrip().split()
    words_lower = [x.lower() for x in words]
    for word in words_lower:
        counts[word_index_dict[word]] += 1

f.close()

# DONE: normalize and writeout counts.
print("Counts before normalization:")
print(counts)

probs = counts / np.sum(counts)
print(probs)


#Step1 : Calculate joint probability of each sentence under unigram model
toy_corpus = open("toy_corpus.txt")
output_file = open("unigram_eval.txt", "w")
for i, line in enumerate(toy_corpus):
    sentence_prob = 1.0
    words = line.strip().split()
    for word in words:
        word_index = word_index_dict.get(word.lower(), -1)
        if word_index != -1:
            word_prob = probs[word_index]
            sentence_prob *= word_prob
    output_file.write(str(sentence_prob) + "\n")

# Verify joint probability of the second sentence
    if i == 1:  # second sentence
        print("Sentence prob =" , sentence_prob)

#Step2: Calculate perplexity of each sentence under unigram model
for i, line in enumerate(toy_corpus):
    sentence_prob = 1.0
    words = line.strip().split()
    for word in words:
        word_index = word_index_dict.get(word.lower(), -1)
        if word_index != -1:
            word_prob = probs[word_index]
            sentence_prob *= word_prob

# Calculate perplexity
sent_len = len(words) + 1  # When adding 1 we account for the end of word token and we get a perplexity of 137.
sent_len_1 = len(words)    # When we don't then we get 153 like in the assignment description.

perplexity = 1/(pow(sentence_prob, 1.0/sent_len))
perplexity_1 = 1/(pow(sentence_prob, 1.0/sent_len_1))

# Write perplexity to output file
output_file.write(str(perplexity) + "\n")

# Verify perplexity of the second sentence
if i == 1:  # second sentence
    print("perplexity when adding 1 :", perplexity)
    print("perplexity when we don't:", perplexity_1)