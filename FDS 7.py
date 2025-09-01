import pandas as pd

# Example order data
data = {
    "customer_id": [101, 102, 101, 103, 102, 101],
    "order_date": pd.to_datetime([
        "2025-01-05", "2025-01-06", "2025-01-10",
        "2025-02-01", "2025-02-05", "2025-02-10"
    ]),
    "product_name": ["Laptop", "Mouse", "Laptop", "Keyboard", "Mouse", "Monitor"],
    "order_quantity": [1, 2, 1, 3, 1, 2]
}

order_data = pd.DataFrame(data)

# 1. Total number of orders made by each customer
orders_per_customer = order_data.groupby("customer_id")["order_date"].count()

# 2. Average order quantity for each product
avg_quantity_per_product = order_data.groupby("product_name")["order_quantity"].mean()

# 3. Earliest and latest order dates
earliest_date = order_data["order_date"].min()
latest_date = order_data["order_date"].max()

# Output
print("Total number of orders by each customer:\n", orders_per_customer)
print("\nAverage order quantity for each product:\n", avg_quantity_per_product)
print("\nEarliest order date:", earliest_date)
print("Latest order date:", latest_date)
