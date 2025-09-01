import pandas as pd

# Example dataset
data = {
    "product_name": ["Laptop", "Mouse", "Keyboard", "Laptop", "Monitor", "Mouse", "Laptop", "Keyboard", "Monitor", "Mouse"],
    "order_quantity": [2, 5, 3, 1, 4, 2, 3, 1, 2, 6],
    "order_date": pd.to_datetime([
        "2025-08-01", "2025-08-02", "2025-08-03", "2025-08-05", "2025-08-07",
        "2025-08-08", "2025-08-10", "2025-08-12", "2025-08-15", "2025-08-20"
    ])
}

sales_data = pd.DataFrame(data)

# ✅ Step 1: (Optional) Filter only the past month if needed
# Example: filter for August 2025
filtered_data = sales_data[sales_data["order_date"].dt.month == 8]

# ✅ Step 2: Group by product and sum the order quantities
product_sales = filtered_data.groupby("product_name")["order_quantity"].sum()

# ✅ Step 3: Get top 5 most sold products
top_5_products = product_sales.sort_values(ascending=False).head(5)

# Output
print("Top 5 most sold products in the past month:\n", top_5_products)
