import numpy as np

# Simulate dataset: 100 patients, 4 symptom features
np.random.seed(42)
X = np.random.rand(100, 4) * 10  # symptom features
y = np.random.choice([0, 1], size=100)  # 0 = no condition, 1 = condition

# User input for new patient
print("Enter symptom values for the new patient (4 features):")
new_patient = []
for i in range(1, 5):
    val = float(input(f"Feature {i}: "))
    new_patient.append(val)

k = int(input("Enter the number of neighbors (k): "))

# Compute Euclidean distances
distances = np.linalg.norm(X - np.array(new_patient), axis=1)

# Get indices of k nearest neighbors
nearest_indices = distances.argsort()[:k]

# Get labels of nearest neighbors
nearest_labels = y[nearest_indices]

# Predict by majority vote
prediction = int(np.round(np.mean(nearest_labels)))
result = "has the condition" if prediction == 1 else "does NOT have the condition"

# Output
print(f"\n🩺 Prediction: The patient {result}.")
print(f"Nearest neighbor votes: {nearest_labels.tolist()}")
