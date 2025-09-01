import pandas as pd
import numpy as np

# Simulate customer ratings (e.g., 1 to 5 stars)
np.random.seed(42)  # for reproducibility
simulated_ratings = np.random.randint(1, 6, size=500)  # 500 ratings between 1 and 5

# Create a DataFrame
df = pd.DataFrame({'rating': simulated_ratings})

# Sample statistics
n = len(df)
mean_rating = df['rating'].mean()
std_dev = df['rating'].std(ddof=1)
std_err = std_dev / np.sqrt(n)

# Confidence level (e.g., 95%)
confidence_level = 0.95
z_table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
z_critical = z_table.get(confidence_level, 1.96)

# Margin of error and confidence interval
margin = z_critical * std_err
ci_lower = mean_rating - margin
ci_upper = mean_rating + margin

# Output results
print(f"📊 Average Rating: {mean_rating:.2f}")
print(f"✅ {int(confidence_level*100)}% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")
print(f"🔍 Margin of Error: ±{margin:.2f}")
