import pandas as pd

# a) Create sample "orders_data"
orders_data = {
    "Order ID": [1, 2, 3, 4, 5, 6],
    "Customer ID": [101, 102, 101, 103, 102, 101],
    "Order Date": [
        "2023-01-01", "2023-01-05", "2023-01-10",
        "2023-02-01", "2023-02-05", "2023-02-15"
    ]
}
orders_df = pd.DataFrame(orders_data)
orders_df["Order Date"] = pd.to_datetime(orders_df["Order Date"])

# b) Create sample "customer_info"
customer_info = {
    "Customer ID": [101, 102, 103],
    "Name": ["Alice", "Bob", "Charlie"],
    "Email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
    "Phone Number": ["111-111-1111", "222-222-2222", "333-333-3333"]
}
customers_df = pd.DataFrame(customer_info)

# c) Merge DataFrames on "Customer ID"
merged_df = pd.merge(orders_df, customers_df, on="Customer ID", how="inner")
print("Merged DataFrame:")
print(merged_df)

# d) Calculate average time between consecutive orders for each customer
merged_df = merged_df.sort_values(by=["Customer ID", "Order Date"])
merged_df["Time_Diff"] = merged_df.groupby("Customer ID")["Order Date"].diff()

avg_time = merged_df.groupby("Customer ID")["Time_Diff"].mean()
overall_avg = avg_time.mean()

print("\nAverage time (per customer):")
print(avg_time)
print(f"\nOverall average time between consecutive orders: {overall_avg}")
