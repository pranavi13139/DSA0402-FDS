import pandas as pd

# Example dataset
data = {
    "property_id": [1, 2, 3, 4, 5],
    "location": ["NYC", "NYC", "LA", "LA", "Chicago"],
    "bedrooms": [3, 5, 4, 6, 2],
    "area_sqft": [1500, 2500, 2000, 3000, 1200],
    "listing_price": [700000, 1200000, 850000, 1500000, 400000]
}

property_data = pd.DataFrame(data)

# 1. Average listing price of properties in each location
avg_price_per_location = property_data.groupby("location")["listing_price"].mean()

# 2. Number of properties with more than 4 bedrooms
properties_more_than_4_bed = property_data[property_data["bedrooms"] > 4].shape[0]

# 3. Property with the largest area
largest_property = property_data.loc[property_data["area_sqft"].idxmax()]

# Output
print("Average listing price per location:\n", avg_price_per_location)
print("\nNumber of properties with more than 4 bedrooms:", properties_more_than_4_bed)
print("\nProperty with the largest area:\n", largest_property)
