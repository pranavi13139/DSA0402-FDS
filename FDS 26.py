import numpy as np

# Simulate concentration data (e.g., 1000 measurements between 2.0 and 5.0 ppm)
np.random.seed(42)  # for reproducibility
population_data = np.random.uniform(low=2.0, high=5.0, size=1000)

# Lookup t-critical values for common confidence levels
t_table = {90: 1.645, 95: 1.96, 99: 2.576}

def confidence_interval(sample, confidence_level):
    n = len(sample)
    mean = np.mean(sample)
    std_dev = np.std(sample, ddof=1)
    std_err = std_dev / np.sqrt(n)
    t_critical = t_table.get(confidence_level, 1.96)  # default to 95%
    margin = t_critical * std_err
    return mean, (mean - margin, mean + margin), margin

def main():
    # User inputs
    sample_size = int(input("Enter sample size: "))
    confidence_level = int(input("Enter confidence level (90, 95, or 99): "))
    precision = float(input("Enter desired level of precision (e.g., 0.5): "))

    if sample_size > len(population_data):
        print("Sample size exceeds available data.")
        return

    # Take random sample
    sample = np.random.choice(population_data, size=sample_size, replace=False)

    # Estimate mean and confidence interval
    mean, ci, margin = confidence_interval(sample, confidence_level)

    print(f"\nEstimated Mean Concentration: {mean:.4f}")
    print(f"{confidence_level}% Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f})")
    print(f"Margin of Error: ±{margin:.4f}")

    if margin <= precision:
        print("✅ Desired precision achieved.")
    else:
        print("⚠️ Desired precision NOT achieved. Consider increasing sample size.")

if __name__ == "__main__":
    main()
