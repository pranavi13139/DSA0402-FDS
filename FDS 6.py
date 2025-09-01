# Example data
prices = [100, 50, 30]        # Prices of items
quantities = [2, 3, 1]        # Quantities of items
discount_rate = 10            # 10% discount
tax_rate = 5                  # 5% tax

# Step 1: Calculate subtotal (sum of price × quantity for each item)
subtotal = sum(p * q for p, q in zip(prices, quantities))

# Step 2: Apply discount
discount_amount = (discount_rate / 100) * subtotal
after_discount = subtotal - discount_amount

# Step 3: Apply tax
tax_amount = (tax_rate / 100) * after_discount
total_cost = after_discount + tax_amount

# Output
print("Subtotal:", subtotal)
print("After Discount:", after_discount)
print("Final Total (with tax):", total_cost)
