import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Create Sample Dataset
# -------------------------------
# Example: 15 students with study time (hrs) and exam scores (%)
study_time = [2, 3, 4, 5, 1, 6, 8, 7, 3, 4, 5, 9, 2, 6, 7]
exam_scores = [50, 55, 60, 65, 40, 70, 85, 80, 58, 62, 67, 90, 48, 72, 78]

df = pd.DataFrame({
    "Study_Hours": study_time,
    "Exam_Score": exam_scores
})

# -------------------------------
# Step 2: Correlation Analysis
# -------------------------------
correlation = df["Study_Hours"].corr(df["Exam_Score"])
print("📊 Correlation between Study Hours and Exam Score:", round(correlation, 3))

# -------------------------------
# Step 3: Visualization
# -------------------------------

plt.figure(figsize=(12,5))

# Scatter plot
plt.subplot(1,3,1)
plt.scatter(df["Study_Hours"], df["Exam_Score"], color="blue")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Scatter Plot: Study Hours vs Exam Score")

# Line plot (trend visualization)
plt.subplot(1,3,2)
plt.plot(df["Study_Hours"], df["Exam_Score"], 'o-', color="green")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Line Plot: Study Hours vs Exam Score")

# Boxplot to check score distribution by study hours
plt.subplot(1,3,3)
df.boxplot(column="Exam_Score", by="Study_Hours", grid=False)
plt.title("Boxplot: Exam Scores by Study Hours")
plt.suptitle("")

plt.tight_layout()
plt.show()
