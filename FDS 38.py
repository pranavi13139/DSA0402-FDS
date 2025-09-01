import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load dataset
# -------------------------------
# Example dataset (replace with your actual CSV)
data = {
    'EngineSize': [1.6, 2.0, 2.4, 3.0, 3.5, 4.0, 4.5],
    'HorsePower': [120, 150, 180, 220, 250, 300, 350],
    'FuelEfficiency': [35, 32, 28, 25, 22, 20, 18],  # mpg
    'Price': [20000, 25000, 28000, 35000, 40000, 45000, 50000]
}
df = pd.DataFrame(data)

# -------------------------------
# Step 2: Prepare features (X) and target (y)
# -------------------------------
X = df[['EngineSize', 'HorsePower', 'FuelEfficiency']].values
y = df['Price'].values.reshape(-1, 1)

# Add bias term (intercept) → column of 1s
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# -------------------------------
# Step 3: Compute Linear Regression using Normal Equation
# -------------------------------
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("Regression Coefficients (theta):")
print(theta)

# Interpreting coefficients
print(f"\nEquation: Price = {theta[0][0]:.2f} + {theta[1][0]:.2f}*EngineSize + {theta[2][0]:.2f}*HorsePower + {theta[3][0]:.2f}*FuelEfficiency")

# -------------------------------
# Step 4: Predictions
# -------------------------------
y_pred = X_b.dot(theta)

# -------------------------------
# Step 5: Model Evaluation
# -------------------------------
ss_total = np.sum((y - y.mean())**2)
ss_residual = np.sum((y - y_pred)**2)
r2 = 1 - (ss_residual / ss_total)

print(f"\nR² Score: {r2:.4f}")

# -------------------------------
# Step 6: Visualization (Actual vs Predicted)
# -------------------------------
plt.scatter(y, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()
