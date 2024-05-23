# Plot the winrate for each init_coin value in a histogram
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("connect_four_init_coin_winrate.csv")
plt.figure(figsize=(6.4, 2.4))
plt.bar(df["init_coin"], df["winrate"])
plt.xlabel("Initial Column of Player 1's first Coin")
plt.ylabel("Winrate Player 2")
plt.tight_layout()
plt.savefig("connect_four_init_coin_winrate.png")
plt.savefig("connect_four_init_coin_winrate.eps")