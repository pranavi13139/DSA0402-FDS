import numpy as np
import matplotlib.pyplot as plt

# Simulate clinical trial data
np.random.seed(42)
control_group = np.random.normal(loc=50, scale=10, size=30)     # placebo group
treatment_group = np.random.normal(loc=60, scale=10, size=30)   # drug group

# Calculate sample statistics
mean_control = np.mean(control_group)
mean_treatment = np.mean(treatment_group)
std_control = np.std(control_group, ddof=1)
std_treatment = np.std(treatment_group, ddof=1)
n_control = len(control_group)
n_treatment = len(treatment_group)

# Welch's t-test (no assumption of equal variances)
se = np.sqrt((std_control**2 / n_control) + (std_treatment**2 / n_treatment))
t_stat = (mean_treatment - mean_control) / se

# Degrees of freedom (Welch-Satterthwaite approximation)
df = ((std_control**2 / n_control + std_treatment**2 / n_treatment)**2) / \
     (((std_control**2 / n_control)**2) / (n_control - 1) + ((std_treatment**2 / n_treatment)**2) / (n_treatment - 1))

# Calculate p-value manually using normal approximation
from math import erf, sqrt
def normal_cdf(x):
    return (1 + erf(x / sqrt(2))) / 2

p_value = 2 * (1 - normal_cdf(abs(t_stat)))  # two-tailed test

# Print results
print(f"Mean (Control): {mean_control:.2f}")
print(f"Mean (Treatment): {mean_treatment:.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"Degrees of freedom: {df:.2f}")
print(f"p-value: {p_value:.4f}")

# Visualization
plt.figure(figsize=(8, 6))
plt.boxplot([control_group, treatment_group], labels=["Placebo", "Treatment"])
plt.title("Clinical Trial Results")
plt.ylabel("Response Level")
plt.grid(True)

# Annotate p-value
plt.text(1.5, max(np.max(control_group), np.max(treatment_group)) + 2,
         f"p-value = {p_value:.4f}", ha='center', fontsize=12, color='blue')

plt.show()
