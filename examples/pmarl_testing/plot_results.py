import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("waterworld_evaders_reward.csv")
plt.scatter(df["n_evaders"], df["n_poisons"], c=df["collected_reward"])
plt.xlabel("Number of evaders", fontsize=20)
plt.ylabel("Number of poisons", fontsize=20)

# Colorbar
cbar = plt.colorbar()
# Colorbar size
cbar.set_label("Collected reward", fontsize=15)

#plt.title("Waterworld evaders reward")
plt.tight_layout()
plt.grid(alpha=0.2)

# Set the font size for the tick labels
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# Legend font size

plt.savefig("waterworld_evaders_reward.png")
plt.savefig("waterworld_evaders_reward.eps")