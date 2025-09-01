import matplotlib.pyplot as plt

# Data
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
rainfall = [120, 85, 90, 150, 200, 175, 130, 100]  # Example values in mm

# Plot
plt.figure(figsize=(8,5))
plt.scatter(months, rainfall, color='blue', s=100)  # s = point size
plt.title("Monthly Rainfall Data")
plt.xlabel("Month")
plt.ylabel("Rainfall (mm)")
plt.grid(True)
plt.show()
