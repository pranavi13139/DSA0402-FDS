import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [2000, 2500, 2200, 2700, 3000, 2800]

plt.figure(figsize=(8,5))
plt.plot(months, sales, marker='o', linestyle='-', color='b')
plt.title("Monthly Sales (Line Plot)")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.show()
