import pandas as pd
from collections import Counter
import re

# Example dataset of customer reviews
data = {
    "review": [
        "The product is really good and useful",
        "Good quality and fast delivery",
        "The product is not worth the price",
        "Excellent product quality and great service",
        "Good product but packaging was bad"
    ]
}

# Create DataFrame
reviews_df = pd.DataFrame(data)

# ✅ Step 1: Combine all reviews into a single text
all_reviews = " ".join(reviews_df["review"].tolist())

# ✅ Step 2: Clean text (lowercase + remove punctuation/numbers)
all_reviews = all_reviews.lower()
all_reviews = re.sub(r'[^a-z\s]', '', all_reviews)  # keep only letters and spaces

# ✅ Step 3: Split into words
words = all_reviews.split()

# ✅ Step 4: Calculate frequency distribution
word_freq = Counter(words)

# ✅ Step 5: Convert to DataFrame for display
freq_df = pd.DataFrame(word_freq.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)

# Display result
print("Frequency Distribution of Words in Customer Reviews:\n")
print(freq_df)
