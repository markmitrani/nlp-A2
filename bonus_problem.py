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

word_counter = {}
bigrams = []
corpus_size = 0
for sentence in brown.sents():
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
    if word_count[1] < 10:
        word_counter_stripped.append(word_count)

word_counter_stripped = np.array(word_counter_stripped)


def find_word_count(word):
    for word_count in word_counter_stripped:
        if word_count[0] == word:
            return word_count[1]
    return "word not found"


def calc_pointwise(word1, word2):
    if word1 in word_counter_stripped and word2 in word_counter_stripped:
        bigrams_count = bigrams.count((word1, word2))
        if bigrams_count != 0:
            w1_count = find_word_count(word1)
            w2_count = find_word_count(word2)
            if isinstance(w1_count, int) and isinstance(w2_count, int):
                N = corpus_size
                return (bigrams_count * N) / (w1_count * w2_count)
            else:
                return "word counter not available for either w1 or w2"
        else:
            return "no bigrams"
    else:
        return "words have less than 10 occurrences"


pointwise_results = []
for bigram in set(bigrams):
    pmi = calc_pointwise(bigram[0], bigram[1])
    if not isinstance(pmi, str):
        pointwise_results.append([bigram, pmi])

pointwise_results = np.array(pointwise_results, dtype=[('name', 'U10'), ('number', int)])
pointwise_results = np.sort(pointwise_results, order='number')

print("TOP 20")
for result in pointwise_results[:20]:
    print(f"{result[0]}: {result[1]}")

print("BOTTOM 20")
for result in pointwise_results[-20:0]:
    print(f"{result[0]}: {result[1]}")
