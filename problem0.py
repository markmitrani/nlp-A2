import nltk
import matplotlib.pyplot as plt
from nltk.corpus import brown
from collections import Counter

###############################
#Finding statistics for Brown Corpus
brown_corpus= brown.words()
num_tokens = len(brown_corpus)
num_types = len(set(brown_corpus))
num_words = len([word for word in brown_corpus if word.isalpha()])
avg_words_per_sentence = num_words/len((brown.sents()))
avg_word_length= sum(len(word) for word in brown_corpus) / num_words
genres = nltk.corpus.brown.categories()

pos_tags = nltk.pos_tag(brown_corpus)
freq_pos_tags = Counter (tag for word, tag in pos_tags).most_common(10)

print("Number of tokens:", num_tokens)
print("Number of types:", num_types)
print("Number of words:", num_words)
print("Average number of words per sentence:", avg_words_per_sentence)
print("Average word length:", avg_word_length)
print("Available genres in the Brown corpus:", genres)
print("Top 10 most frequent POS tags:")

for tag, freq in freq_pos_tags:
    print(tag, "-", freq)

###################################

#Question(I): Computing a list of unique words for the whole corpus by descending frequency:
def compute_freq_dist(words):
    freq_dist = nltk.FreqDist(words)
    return sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)
freq_dist_corpus = compute_freq_dist(brown_corpus)

####################################

#Question(II): Computing a list of unique words for genres (learned and humor) by descending frequency:

learned_words = brown.words(categories='learned')
humor_words = brown.words(categories='humor')
freq_dist_learned = compute_freq_dist(learned_words)
freq_dist_humor = compute_freq_dist(humor_words)

#########################

#Plotting frequencies:
#1: Linear axis:
plt.figure(figsize=(10, 5))
plt.plot([freq for word, freq in freq_dist_corpus])
plt.title("Whole Corpus")
plt.xlabel("Position in frequency list")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot([freq for word, freq in freq_dist_learned])
plt.title("Learned-Corpus")
plt.xlabel("Position in frequency list")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot([freq for word, freq in freq_dist_humor])
plt.title("Humor-Corpus")
plt.xlabel("Position in frequency list")
plt.ylabel("Frequency")
plt.show()

########################
#2: log-log Axis:
plt.figure(figsize=(10, 5))
plt.plot([freq for word, freq in freq_dist_corpus])
plt.xscale('log')
plt.yscale('log')
plt.title("Whole Corpus")
plt.xlabel("Position in frequency list")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot([freq for word, freq in freq_dist_learned])
plt.xscale('log')
plt.yscale('log')
plt.title("Learned-Corpus")
plt.xlabel("Position in frequency list")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot([freq for word, freq in freq_dist_humor])
plt.xscale('log')
plt.yscale('log')
plt.title("Humor-Corpus")
plt.xlabel("Position in frequency list")
plt.ylabel("Frequency")
plt.show()


