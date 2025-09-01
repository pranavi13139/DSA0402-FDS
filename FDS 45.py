import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Create dataset manually
# -------------------------------
data = {
    "Name": ["Messi", "Ronaldo", "Neymar", "Mbappe", "Lewandowski",
             "Salah", "De Bruyne", "Kane", "Benzema", "Modric"],
    "Age": [36, 38, 31, 25, 34, 30, 32, 29, 35, 37],
    "Position": ["Forward", "Forward", "Forward", "Forward", "Forward",
                 "Forward", "Midfielder", "Forward", "Forward", "Midfielder"],
    "Goals": [25, 22, 18, 30, 27, 20, 10, 21, 24, 5],
    "WeeklySalary": [1000000, 950000, 700000, 850000, 800000,
                     650000, 600000, 620000, 750000, 400000]
}

df = pd.DataFrame(data)

# Save dataset to CSV
df.to_csv("soccer_players.csv", index=False)

# -------------------------------
# Step 2: Read from CSV
# -------------------------------
df = pd.read_csv("soccer_players.csv")

# -------------------------------
# Step 3: Top 5 players by Goals
# -------------------------------
top_goals = df.nlargest(5, "Goals")[["Name", "Goals"]]
print("Top 5 Players by Goals:")
print(top_goals, "\n")

# -------------------------------
# Step 4: Top 5 players by Salary
# -------------------------------
top_salary = df.nlargest(5, "WeeklySalary")[["Name", "WeeklySalary"]]
print("Top 5 Players by Salary:")
print(top_salary, "\n")

# -------------------------------
# Step 5: Average Age and Players above average
# -------------------------------
avg_age = df["Age"].mean()
above_avg = df[df["Age"] > avg_age][["Name", "Age"]]
print(f"Average Age of Players: {avg_age:.2f}")
print("Players above average age:")
print(above_avg, "\n")

# -------------------------------
# Step 6: Visualization - Bar chart by Position
# -------------------------------
position_counts = df["Position"].value_counts()

plt.bar(position_counts.index, position_counts.values, color="skyblue")
plt.title("Distribution of Players by Position")
plt.xlabel("Position")
plt.ylabel("Number of Players")
plt.show()
