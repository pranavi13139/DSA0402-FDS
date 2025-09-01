import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Example dataset
# (You can replace this with pd.read_csv("transactions.csv"))
# -------------------------------
data = {
    "CustomerID": range(1, 11),
    "TotalSpent": [500, 1500, 700, 200, 3000, 1200, 400, 3500, 800, 600],
    "ItemsPurchased": [5, 12, 6, 2, 25, 15, 3, 30, 8, 4]
}
df = pd.DataFrame(data)

# -------------------------------
# Step 2: Prepare features
# -------------------------------
X = df[["TotalSpent", "ItemsPurchased"]].values

# -------------------------------
# Step 3: Implement K-Means from scratch
# -------------------------------
def kmeans(X, k=3, max_iters=100):
    # randomly choose initial centroids
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # assign clusters based on closest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # convergence check
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

# Run KMeans
k = 3  # number of clusters
labels, centroids = kmeans(X, k)

# Add labels to dataframe
df["Cluster"] = labels

# -------------------------------
# Step 4: Visualization
# -------------------------------
colors = ["red", "blue", "green"]
for i in range(k):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                c=colors[i], label=f"Cluster {i}", s=100, alpha=0.6)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], 
            c="black", marker="X", s=200, label="Centroids")

plt.xlabel("Total Spent")
plt.ylabel("Items Purchased")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Step 5: Insights
# -------------------------------
print("Clustered Data:")
print(df)
