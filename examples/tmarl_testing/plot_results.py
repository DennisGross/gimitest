# Plot the winrate for each init_coin value in a histogram
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
df = pd.read_csv("connect_four_init_coin_winrate.csv")

# Create a figure with specific size
plt.figure(figsize=(6.4, 3))

# Plot the bar chart
plt.bar(df["init_coin"], df["winrate"])

# Set labels for the axes
plt.xlabel("Initial Column of Player 1's first Coin", fontsize=20)
plt.ylabel("Winrate Player 2", fontsize=20)

# Set the font size for the tick labels
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Ensure the layout is tight
plt.tight_layout()

# Save the figure in different formats
plt.savefig("connect_four_init_coin_winrate.png")
plt.savefig("connect_four_init_coin_winrate.eps")
