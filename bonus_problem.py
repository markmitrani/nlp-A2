# FORMULA:
# C(wordpair)*N / C(w1) * C(w2)
# count(word pair)
# count word1
# count word2
# N = corpus word size


# keep in mind
# Words (not word pairs) that occur in the corpus less than 10 times
# should be ignored.

# output
# list the 20 word pairs with the highest pmi value
# the 20 word pairs with the lowest pmi value.

from nltk.corpus import brown
import numpy as np
from tqdm import tqdm
from collections import Counter

word_counter = {}
bigrams = []
corpus_size = 0
for sentence in tqdm(brown.sents(), desc="processing brown corpus"):
    sentence = " ".join(sentence)
    sentence = sentence.lower().rstrip().split()
    previous_word = sentence[0]
    corpus_size += 1
    if sentence[0] not in word_counter.keys():
        word_counter.setdefault(sentence[0], 1)
    else:
        word_counter[sentence[0]] += 1

    for word in sentence[1:]:
        corpus_size += 1
        bigrams.append((previous_word, word))
        if word not in word_counter.keys():
            word_counter.setdefault(word, 1)
        else:
            word_counter[word] += 1
        previous_word = word

word_counter_stripped = []

for word_count in word_counter.items():
    if word_count[1] >= 10:
        word_counter_stripped.append(word_count)

word_counter_stripped = dict(word_counter_stripped)

'''
# slow af
def find_word_count(word):
    for word_count in word_counter_stripped:
        if word_count[0] == word:
            return word_count[1]
    return "word not found"
'''

bigrams_counts = Counter(bigrams)


def calc_pointwise(word1, word2):
    if word1 in word_counter_stripped and word2 in word_counter_stripped:
        # bigrams_count = bigrams.count((word1, word2))
        bigrams_count = bigrams_counts[(word1, word2)]
        if bigrams_count != 0:
            if word1 in word_counter_stripped.keys() and word2 in word_counter_stripped.keys():
                w1_count = word_counter_stripped[word1]
                w2_count = word_counter_stripped[word2]
                N = corpus_size
                return (bigrams_count * N) / (w1_count * w2_count)
            else:
                return "word counter not available for either w1 or w2"
        else:
            return "no bigrams"
    else:
        return "words have less than 10 occurrences"


pointwise_results = []
for bigram in tqdm(set(bigrams), desc="processing bigrams"):
    pmi = calc_pointwise(bigram[0], bigram[1])
    if not isinstance(pmi, str):
        pointwise_results.append([bigram, pmi])

print(pointwise_results[0])
# pointwise_results = np.array(pointwise_results, dtype=[('words', 'U10,U10', (2,)), ('pmi', float)])
# pointwise_results = np.sort(pointwise_results, order='pmi')
pointwise_results = sorted(pointwise_results, key=lambda x: x[1], reverse=True)

top_file = open("bonus_top.txt", "w")
print("TOP 20")
top_file.write("TOP 20\n")
for result in pointwise_results[:20]:
    print(f"{result[0]}: {result[1]}")
    top_file.write(f"{result[0]}: {result[1]}\n")


bot_file = open("bonus_bot.txt", "w")
print("BOTTOM 20")
bot_file.write("BOTTOM 20\n")
for result in pointwise_results[-20:]:
    print(f"{result[0]}: {result[1]}")
    bot_file.write(f"{result[0]}: {result[1]}\n")

print(f"-20: {pointwise_results[-20]}, -1: {pointwise_results[-1]}")

top_file.close()
bot_file.close()
