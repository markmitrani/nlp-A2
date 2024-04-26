#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
import codecs

# PROBLEM 2 START
vocab = open("brown_vocab_100.txt")
word_index_dict = {}
for i, line in enumerate(vocab):
    word = line.rstrip()
    word_index_dict[word] = i
f = open("brown_100.txt")
counts = np.zeros(len(word_index_dict))
for i, line in enumerate(f):
    words = line.rstrip().split()
    words_lower = [x.lower() for x in words]
    for word in words_lower:
        counts[word_index_dict[word]] += 1
f.close()
probs = counts / np.sum(counts)
# PROBLEM 2 END

# PROBLEM 6 for unigram

unigram = probs
toy_sentences = codecs.open("toy_corpus.txt")
result_txt = open("unigram_eval.txt","w")
for toy_sentence in toy_sentences:
    print(f"for sentence: {toy_sentence}")
    toy_sentence = toy_sentence.lower().rstrip().split()
    sentprob = 1
    sent_len = len(toy_sentence)
    for toy_word in toy_sentence:
        sentprob *= unigram[word_index_dict[toy_word]]
    perplexity = 1 / (pow(sentprob, 1.0 / sent_len))
    print(f"we get prob: {sentprob}")
    print(f"we get perplexity: {perplexity}")
    result_txt.write(f"{perplexity}\n")


result_txt.close()

