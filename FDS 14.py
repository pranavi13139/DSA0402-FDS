import pandas as pd

# Example sales dataset
data = {
    "customer_id": [101, 102, 103, 104, 105, 106, 107],
    "age": [25, 30, 22, 25, 40, 30, 25],
    "purchase_amount": [200, 150, 300, 100, 250, 400, 180]
}

sales_data = pd.DataFrame(data)

# ✅ Step 1: Frequency distribution of ages
age_distribution = sales_data["age"].value_counts().sort_index()

# ✅ Step 2: Display result
print("Frequency Distribution of Customer Ages:\n")
print(age_distribution)
