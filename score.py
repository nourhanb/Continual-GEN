import random

# Generate 144 random values
x = [random.uniform(0, 2) for _ in range(5)]
print(sorted(x))
print(sorted(x, reverse=False)[:2])  
# Define the ensemble strategies
# Define the ensemble strategies with sorting in descending order
strategies = [
    ("Top 20", lambda x: sorted(x, reverse=False)[:2]),
    ("Top 40", lambda x: sorted(x, reverse=False)[:3]),
    ("Bottom 20", lambda x: sorted(x, reverse=True)[:2]),
    ("Bottom 40", lambda x: sorted(x, reverse=True)[:3])
]


# Apply ensemble strategies and calculate averages
for strategy_name, strategy in strategies:
    selected_values = strategy(x)
    average = sum(selected_values) / len(selected_values)
    print(f"{strategy_name}: {average}")