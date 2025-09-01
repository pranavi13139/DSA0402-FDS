import numpy as np

# Simulate customer data
np.random.seed(42)
purchase_freq = np.random.randint(1, 20, size=100)
avg_spend = np.random.uniform(100, 1000, size=100)
time_on_site = np.random.uniform(5, 60, size=100)

X = np.column_stack((purchase_freq, avg_spend, time_on_site))

# K-Means parameters
k = 3
max_iters = 100

# Initialize centroids randomly
centroids = X[np.random.choice(len(X), k, replace=False)]

# K-Means algorithm
for _ in range(max_iters):
    # Assign clusters
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    # Update centroids
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    
    # Check for convergence
    if np.allclose(centroids, new_centroids):
        break
    centroids = new_centroids

# User input for new customer
print("Enter shopping-related features of the new customer:")
freq_input = float(input("Purchase frequency (per month): "))
spend_input = float(input("Average spend per purchase (₹): "))
time_input = float(input("Average time on site (minutes): "))

new_customer = np.array([freq_input, spend_input, time_input])

# Assign new customer to nearest cluster
distances_to_centroids = np.linalg.norm(centroids - new_customer, axis=1)
assigned_cluster = np.argmin(distances_to_centroids)

# Output
print(f"\n🧠 The new customer belongs to Segment {assigned_cluster}")
