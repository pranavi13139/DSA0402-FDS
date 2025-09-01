import numpy as np

# Sample conversion rate data (replace with actual data)
# These could be percentages or proportions (e.g., 0.12 for 12%)
design_A = [0.10, 0.12, 0.11, 0.09, 0.13, 0.14, 0.10, 0.11, 0.12, 0.13]
design_B = [0.15, 0.14, 0.16, 0.13, 0.17, 0.18, 0.14, 0.15, 0.16, 0.17]

# Calculate sample statistics
mean_A = np.mean(design_A)
mean_B = np.mean(design_B)
std_A = np.std(design_A, ddof=1)
std_B = np.std(design_B, ddof=1)
n_A = len(design_A)
n_B = len(design_B)

# Calculate pooled standard error
se = np.sqrt((std_A**2 / n_A) + (std_B**2 / n_B))

# Degrees of freedom (approximate using Welch-Satterthwaite equation)
df = ((std_A**2 / n_A + std_B**2 / n_B)**2) / \
     (((std_A**2 / n_A)**2) / (n_A - 1) + ((std_B**2 / n_B)**2) / (n_B - 1))

# Calculate t-statistic
t_stat = (mean_A - mean_B) / se

# Manually set critical t-value for 95% confidence (two-tailed)
# For df ≈ 18, t_critical ≈ 2.101
t_critical = 2.101

# Decision
if abs(t_stat) > t_critical:
    print("Statistically significant difference in conversion rates.")
else:
    print("No statistically significant difference in conversion rates.")

# Optional: print details
print(f"Mean A: {mean_A:.4f}, Mean B: {mean_B:.4f}")
print(f"t-statistic: {t_stat:.4f}, Degrees of freedom: {df:.2f}")
