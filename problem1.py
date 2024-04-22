#!/usr/bin/env python3
"""
NLP A2: N-Gram Language Models

@author: Dany, Markito, El Samo

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

word_index_dict = {}

# DONE: read brown_vocab_100.txt into word_index_dict
with open("brown_vocab_100.txt", "r") as file:
    count = 0
    for line in file:
        word_index_dict.setdefault(line.rstrip(), count)
        count += 1
# DONE: write word_index_dict to word_to_index_100.txt
with open("word_to_index_100.txt", "w") as file:
    file.write(str(word_index_dict))

print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))
