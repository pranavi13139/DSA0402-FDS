import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Example Data ----
data = {
    'Age': [23, 25, 27, 32, 34, 36, 41, 43, 46, 48, 52, 54, 57, 59, 61, 64, 66, 68],
    'Fat': [9.5, 10.1, 10.8, 12.3, 13.5, 14.2, 15.8, 16.4, 17.2, 18.9, 
            20.3, 21.1, 22.8, 23.7, 24.5, 25.1, 26.4, 27.8]
}

df = pd.DataFrame(data)

# ---- Descriptive Statistics ----
print("Mean values:\n", df.mean())
print("\nMedian values:\n", df.median())
print("\nStandard Deviation:\n", df.std())

# ---- Boxplots ----
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.boxplot(df['Age'])
plt.title("Boxplot of Age")
plt.ylabel("Age")

plt.subplot(1,2,2)
plt.boxplot(df['Fat'])
plt.title("Boxplot of %Fat")
plt.ylabel("%Fat")

plt.show()

# ---- Scatter Plot ----
plt.figure(figsize=(6,5))
plt.scatter(df['Age'], df['Fat'], color="blue")
plt.xlabel("Age")
plt.ylabel("%Fat")
plt.title("Scatter Plot of Age vs %Fat")
plt.show()

# ---- Q-Q Plot (Manual with NumPy + Matplotlib) ----
def qqplot(data, title):
    data = np.sort(data)
    n = len(data)
    # theoretical quantiles from a normal distribution
    theoretical_q = np.sort(np.random.normal(np.mean(data), np.std(data), n))
    
    plt.scatter(theoretical_q, data, color="purple")
    min_val = min(min(theoretical_q), min(data))
    max_val = max(max(theoretical_q), max(data))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # 45° line
    plt.title(title)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
qqplot(df['Age'], "Q-Q Plot of Age")

plt.subplot(1,2,2)
qqplot(df['Fat'], "Q-Q Plot of %Fat")

plt.show()
