import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("waterworld_evaders_reward.csv")
plt.scatter(df["n_evaders"], df["n_poisons"], c=df["collected_reward"])
plt.xlabel("Number of evaders")
plt.ylabel("Number of poisons")
# Colorbar
cbar = plt.colorbar()
cbar.set_label("Collected reward")
#plt.title("Waterworld evaders reward")
plt.tight_layout()
plt.grid(alpha=0.2)
plt.savefig("waterworld_evaders_reward.png")
plt.savefig("waterworld_evaders_reward.eps")