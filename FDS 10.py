import matplotlib.pyplot as plt

# Example dataset
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
sales = [2500, 3000, 2800, 3500, 4000, 3700, 4200, 4500]

# -------------------------------
# 1. Line Plot
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(months, sales, marker='o', linestyle='-', color='b')
plt.title("Monthly Sales Data (Line Plot)")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# -------------------------------
# 2. Bar Plot
# -------------------------------
plt.figure(figsize=(8,5))
plt.bar(months, sales, color='orange')
plt.title("Monthly Sales Data (Bar Plot)")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()
