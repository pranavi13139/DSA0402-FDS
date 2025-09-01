import numpy as np

# Example dataset: [bedrooms, square_footage, sale_price]
house_data = np.array([
    [3, 1800, 250000],
    [5, 2500, 400000],
    [4, 2000, 320000],
    [6, 3000, 500000],
    [2, 1500, 200000]
])

# Step 1: Filter rows where bedrooms > 4
houses_with_more_than_4 = house_data[house_data[:, 0] > 4]

# Step 2: Extract the sale_price column (last column)
sale_prices = houses_with_more_than_4[:, -1]

# Step 3: Calculate average sale price
average_price = np.mean(sale_prices)

# Output
print("Average sale price of houses with more than 4 bedrooms:", average_price)
