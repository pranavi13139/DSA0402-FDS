import numpy as np

# Example 4x4 student scores (rows = students, columns = subjects)
# Subjects order: Math, Science, English, History
student_scores = np.array([
    [85, 90, 78, 92],   # Student 1
    [88, 76, 85, 80],   # Student 2
    [90, 92, 88, 84],   # Student 3
    [70, 80, 75, 85]    # Student 4
])

# Step 1: Calculate average score for each subject (column-wise mean)
subject_avg = np.mean(student_scores, axis=0)

# Step 2: Define subject names
subjects = ["Math", "Science", "English", "History"]

# Step 3: Find subject with highest average
highest_index = np.argmax(subject_avg)
highest_subject = subjects[highest_index]

# Output
print("Average scores for each subject:")
for sub, avg in zip(subjects, subject_avg):
    print(f"{sub}: {avg:.2f}")

print("\nSubject with the highest average score:", highest_subject)
