import pandas as pd
import matplotlib.pyplot as plt

# a) Create a sample DataFrame
data = {
    "Date": pd.date_range(start="2023-01-01", periods=90, freq="D"),
    "Temperature (Celsius)": [20 + i % 10 + (i*0.1) for i in range(90)]
}
df = pd.DataFrame(data)

# b) Convert "Date" column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# c) Calculate average temperature per month
df["Month"] = df["Date"].dt.to_period("M")
monthly_avg = df.groupby("Month")["Temperature (Celsius)"].mean().reset_index()

print("Average Monthly Temperatures:")
print(monthly_avg)

# d) Plot temperature trend over time
plt.figure(figsize=(10,5))
plt.plot(df["Date"], df["Temperature (Celsius)"], marker="o", linestyle="-", label="Daily Temperature")
plt.title("Temperature Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (Celsius)")
plt.legend()
plt.grid(True)
plt.show()
