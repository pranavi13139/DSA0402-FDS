import pandas as pd

# ---- Sample sales dataset (instead of reading CSV) ----
data = {
    "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    "Product": ["Laptop", "Phone", "Tablet", "Laptop", "Headphones"],
    "Quantity Sold": [5, 10, 8, 3, 15],
    "Unit Price": [600, 300, 200, 600, 50]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# b) Create "Total Sales" column
df["Total Sales"] = df["Quantity Sold"] * df["Unit Price"]

# c) Calculate total sales for each product
product_sales = df.groupby("Product")["Total Sales"].sum().reset_index()

# Add profit column (20% margin)
product_sales["Profit"] = product_sales["Total Sales"] * 0.20

# Sort by Profit (descending)
top_products = product_sales.sort_values(by="Profit", ascending=False)

# Display top 5 most profitable products
print("Top 5 Most Profitable Products:\n")
print(top_products.head(5))

# Overall profit
overall_profit = product_sales["Profit"].sum()
print("\nOverall Profit (20% margin):", overall_profit)
