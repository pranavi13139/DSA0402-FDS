import pandas as pd

# ---- Sample customer dataset ----
data = {
    "Customer ID": [101, 102, 103, 104, 105, 106, 107, 108],
    "Age": [25, 34, 29, 45, 52, 40, 30, 60],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
    "Total Spending": [1200, 800, 300, 2000, 1500, 600, 400, 2500]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# ---- b) Segment customers based on Total Spending ----
def spending_segment(spending):
    if spending >= 1500:
        return "High Spender"
    elif spending >= 700:
        return "Medium Spender"
    else:
        return "Low Spender"

df["Segment"] = df["Total Spending"].apply(spending_segment)

# ---- c) Calculate average age per segment ----
avg_age = df.groupby("Segment")["Age"].mean().reset_index()

# ---- Results ----
print("Customer Data with Segments:\n")
print(df)

print("\nAverage Age by Segment:\n")
print(avg_age)
