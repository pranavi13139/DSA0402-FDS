import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load dataset
# -------------------------------
# Example dataset (replace with your CSV data)
data = {
    'HouseSize': [850, 900, 1200, 1500, 1700, 2000, 2200, 2500],
    'Price':     [150000, 160000, 200000, 250000, 280000, 320000, 350000, 400000]
}
df = pd.DataFrame(data)

X = df['HouseSize'].values
y = df['Price'].values

# -------------------------------
# Step 2: Compute Linear Regression (Y = b0 + b1*X)
# -------------------------------
# Mean of X and Y
X_mean = np.mean(X)
y_mean = np.mean(y)

# Calculate slope (b1) and intercept (b0)
b1 = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)
b0 = y_mean - b1 * X_mean

print(f"Regression Equation: Price = {b0:.2f} + {b1:.2f}*HouseSize")

# -------------------------------
# Step 3: Predictions
# -------------------------------
y_pred = b0 + b1 * X

# -------------------------------
# St
