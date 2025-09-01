import string
from collections import Counter
import matplotlib.pyplot as plt

# Instead of reading from file
text = """This is a sample text document. 
This document is just for testing word frequency distribution. 
This sample is simple and useful for text analysis."""

# Preprocess
text = text.lower()
text = text.translate(str.maketrans("", "", string.punctuation))
words = text.split()

# Count frequencies
word_freq = Counter(words)

print("Word Frequency Distribution:")
for word, freq in word_freq.items():
    print(f"{word}: {freq}")

# Plot top 10 words
top_words = word_freq.most_common(10)
words_list, freq_list = zip(*top_words)

plt.figure(figsize=(8,5))
plt.bar(words_list, freq_list, color='skyblue')
plt.title("Top 10 Most Frequent Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()
