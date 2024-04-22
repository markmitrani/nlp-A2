#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""
word_index_dict = {}

# read brown_vocab_100.txt into word_index_dict
with open("brown_vocab_100.txt", "r") as file:
    for i, line in enumerate(file.readlines(), 0):
        word = line.rstrip()
        word_index_dict[word] = i

# write word_index_dict to word_to_index_100.txt
with open("word_to_index_100.txt", "w") as file:
    dict_str = str(word_index_dict)
    file.write(dict_str)

print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))
