import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function (binary cross-entropy)
def compute_cost(X, y, weights):
    m = len(y)
    predictions = sigmoid(X @ weights)
    cost = -1/m * np.sum(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))
    return cost

# Gradient descent
def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    for _ in range(epochs):
        predictions = sigmoid(X @ weights)
        gradient = (1/m) * (X.T @ (predictions - y))
        weights -= lr * gradient
    return weights

# Simulate dataset: [usage_minutes, contract_duration]
np.random.seed(42)
usage = np.random.randint(100, 1000, size=100)
contract = np.random.randint(1, 24, size=100)
X_raw = np.column_stack((usage, contract))

# Simulated churn labels: 0 = not churned, 1 = churned
y = ((usage < 400) & (contract < 6)).astype(int)

# Normalize features
X_norm = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)

# Add bias term
X = np.column_stack((np.ones(X_norm.shape[0]), X_norm))

# Train model
weights = train_logistic_regression(X, y)

# User input
print("Enter details of the new customer:")
usage_input = float(input("Usage minutes per month: "))
contract_input = int(input("Contract duration (in months): "))

# Normalize input
usage_norm = (usage_input - np.mean(usage)) / np.std(usage)
contract_norm = (contract_input - np.mean(contract)) / np.std(contract)
new_customer = np.array([1, usage_norm, contract_norm])

# Predict
probability = sigmoid(new_customer @ weights)
prediction = int(probability >= 0.5)
status = "will churn" if prediction == 1 else "will NOT churn"

# Output
print(f"\n📊 Prediction: The customer {status}.")
print(f"🔍 Churn probability: {probability:.2f}")
