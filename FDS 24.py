import numpy as np

# Sample data (replace with your actual values)
drug_group = [12, 14, 11, 13, 15, 10, 12, 13, 14, 11, 12, 13, 14, 15, 11, 12, 13, 14, 10, 11, 12, 13, 14, 15, 13]
placebo_group = [4, 5, 3, 6, 2, 4, 5, 3, 4, 6, 2, 3, 4, 5, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 2]

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # sample standard deviation
    std_err = std_dev / np.sqrt(n)

    # For 95% CI and n=25, t-critical ≈ 2.064 (from t-distribution table)
    # You can adjust this manually for different sample sizes or confidence levels
    t_critical = 2.064

    margin = t_critical * std_err
    return (mean - margin, mean + margin)

# Calculate 95% confidence intervals
drug_ci = confidence_interval(drug_group)
placebo_ci = confidence_interval(placebo_group)

print(f"95% CI for Drug Group: {drug_ci}")
print(f"95% CI for Placebo Group: {placebo_ci}")
