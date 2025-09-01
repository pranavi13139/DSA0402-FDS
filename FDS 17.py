import pandas as pd

# Create sample CSV with a 'feedback' column
data = {
    "feedback": [
        "The product quality is excellent and delivery was fast",
        "Customer service was very helpful and polite",
        "The price is reasonable but shipping was delayed",
        "Excellent product, I will recommend to others",
        "Not satisfied with the quality, needs improvement"
    ]
}

df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)
print("Sample data.csv file created successfully!")
