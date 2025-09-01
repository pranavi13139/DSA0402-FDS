import pandas as pd
import numpy as np

# -------------------------------
# Step 1: Sample Dataset (can be replaced with CSV file)
# -------------------------------
# Example: Daily temperature readings (°C) for 3 cities
data = {
    "City": ["Delhi"]*7 + ["Mumbai"]*7 + ["Bangalore"]*7,
    "Temperature": [15, 18, 20, 22, 25, 28, 30,     # Delhi
                    28, 29, 30, 31, 32, 33, 34,     # Mumbai
                    20, 21, 22, 23, 24, 25, 26]     # Bangalore
}

df = pd.DataFrame(data)

# -------------------------------
# Step 2: Calculations
# -------------------------------

# a) Mean temperature for each city
mean_temp = df.groupby("City")["Temperature"].mean()

# b) Standard deviation for each city
std_temp = df.groupby("City")["Temperature"].std()

# c) City with highest temperature range
temp_range = df.groupby("City")["Temperature"].apply(lambda x: x.max() - x.min())
highest_range_city = temp_range.idxmax()

# d) City with most consistent temperature (lowest std dev)
consistent_city = std_temp.idxmin()

# -------------------------------
# Step 3: Display Results
# -------------------------------
print("📊 Mean Temperature (°C):")
print(mean_temp, "\n")

print("📊 Standard Deviation of Temperature (°C):")
print(std_temp, "\n")

print("🌡️ City with Highest Temperature Range:", highest_range_city, "(", temp_range[highest_range_city], "°C )")
print("❄️ Most Consistent City (Lowest Std Dev):", consistent_city, "(", round(std_temp[consistent_city], 2), "°C )")
