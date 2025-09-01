import pandas as pd

# Example dataset: post_id and number of likes
data = {
    "post_id": [1, 2, 3, 4, 5, 6, 7],
    "likes": [10, 25, 10, 40, 25, 10, 50]
}

# Create DataFrame
posts_data = pd.DataFrame(data)

# ✅ Step 1: Calculate frequency distribution of likes
likes_distribution = posts_data["likes"].value_counts().sort_index()

# ✅ Step 2: Display result
print("Frequency Distribution of Likes among Posts:\n")
print(likes_distribution)
