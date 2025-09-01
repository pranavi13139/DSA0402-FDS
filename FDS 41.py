import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------
# Step 1: Load stock data safely
# -------------------------------
file_path = "stock_data.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("✅ Stock data loaded from CSV file.")
else:
    print("⚠️ stock_data.csv not found. Generating sample data...")
    # Generate sample stock data
    dates = pd.date_range(start="2023-01-01", periods=100)
    prices = np.cumsum(np.random.randn(100)) + 150  # random walk around 150
    df = pd.DataFrame({"Date": dates, "Close": prices})

# Ensure Date is datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# -------------------------------
# Step 2: Calculate Variability
# -------------------------------
closing_prices = df['Close']

mean_price = closing_prices.mean()
variance = closing_prices.var()
std_dev = closing_prices.std()

# Daily returns (% change)
df['Daily_Return'] = closing_prices.pct_change()

# Volatility = standard deviation of daily returns
volatility = df['Daily_Return'].std()

# -------------------------------
# Step 3: Print Insights
# -------------------------------
print("\n📊 Stock Price Variability Analysis")
print(f"Average Closing Price: {mean_price:.2f}")
print(f"Variance of Prices: {variance:.2f}")
print(f"Standard Deviation of Prices: {std_dev:.2f}")
print(f"Volatility (std of daily returns): {volatility:.4f}")

# -------------------------------
# Step 4: Visualization
# -------------------------------
plt.figure(figsize=(12,5))

# Plot closing prices
plt.subplot(1,2,1)
plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Closing Prices Over Time")
plt.legend()

# Plot daily returns
plt.subplot(1,2,2)
plt.plot(df['Date'], df['Daily_Return'], label='Daily Return', color='green')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.title("Daily Returns")
plt.legend()

plt.tight_layout()
plt.show()
