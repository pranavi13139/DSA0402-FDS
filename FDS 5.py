import numpy as np

# Example dataset: fuel efficiency (MPG) of car models
fuel_efficiency = np.array([20, 25, 30, 35])

# Step 1: Calculate average fuel efficiency
average_efficiency = np.mean(fuel_efficiency)

# Step 2: Select two car models (e.g., model 1 and model 4)
model_a = fuel_efficiency[0]   # First car model
model_b = fuel_efficiency[3]   # Fourth car model

# Step 3: Calculate percentage improvement from model A → model B
percentage_improvement = ((model_b - model_a) / model_a) * 100

# Output
print("Average fuel efficiency:", average_efficiency, "MPG")
print(f"Percentage improvement from Model A to Model B: {percentage_improvement:.2f}%")
