import numpy as np

# Example 3x3 sales data (rows = products, columns = sales prices)
sales_data = np.array([
    [200, 220, 250],   # Product 1 sales prices
    [150, 180, 170],   # Product 2 sales prices
    [300, 310, 290]    # Product 3 sales prices
])

# Step 1: Calculate the overall average price
average_price = np.mean(sales_data)

# Output
print("Average price of all products sold in the past month:", average_price)
