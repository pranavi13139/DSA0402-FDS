import numpy as np

# Example sales data for 4 quarters
sales_data = np.array([100000, 120000, 150000, 180000])

# Step 1: Total sales for the year
total_sales = np.sum(sales_data)

# Step 2: Percentage increase from Q1 to Q4
q1_sales = sales_data[0]
q4_sales = sales_data[3]
percentage_increase = ((q4_sales - q1_sales) / q1_sales) * 100

# Output
print("Total sales for the year:", total_sales)
print("Percentage increase from Q1 to Q4:", percentage_increase, "%")
