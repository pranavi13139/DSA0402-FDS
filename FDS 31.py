import numpy as np

# Simulate dataset: [area (sq ft), bedrooms, location index]
np.random.seed(42)
area = np.random.randint(500, 3000, size=100)
bedrooms = np.random.randint(1, 5, size=100)
location = np.random.randint(0, 3, size=100)  # 0 = rural, 1 = suburban, 2 = urban

# Stack features into matrix X and add bias term
X = np.column_stack((area, bedrooms, location))
X = np.column_stack((np.ones(X.shape[0]), X))  # add intercept term

# Simulate prices (in thousands)
y = area * 0.3 + bedrooms * 50 + location * 100 + np.random.normal(0, 100, size=100)

# Compute weights using Normal Equation: w = (XᵀX)⁻¹Xᵀy
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
XtY = X.T @ y
weights = XtX_inv @ XtY

# User input for new house
print("Enter details of the new house:")
area_input = float(input("Area (in sq ft): "))
bedrooms_input = int(input("Number of bedrooms: "))
location_input = int(input("Location index (0 = rural, 1 = suburban, 2 = urban): "))

# Prepare input vector with bias term
new_house = np.array([1, area_input, bedrooms_input, location_input])
predicted_price = new_house @ weights

# Output
print(f"\n🏡 Predicted price of the house: ₹{predicted_price:.2f}K")
