import pandas as pd

# a) Create a sample DataFrame
data = {
    "Employee ID": [101, 102, 103, 104],
    "Full Name": ["Alice Johnson", "Bob Smith", "Charlie Brown", "David Lee"],
    "Department": ["HR", None, "IT", "Finance"],
    "Salary": ["50000", "60000", "abc", "70000"]
}

df = pd.DataFrame(data)

# b) Convert "Salary" column to numeric
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

# c) Remove rows with missing "Department"
df = df[df["Department"].notna()]

# d) Extract first name from "Full Name"
df["First Name"] = df["Full Name"].apply(lambda x: str(x).split()[0] if pd.notna(x) else "")

# Show final cleaned DataFrame
print(df)
