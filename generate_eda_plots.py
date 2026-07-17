import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data\engineered_energy_dataset.csv")

plt.style.use("default")
sns.set_palette("Blues_d")

# 1. Energy usage by hour
plt.figure(figsize=(8,5))
hourly = df.groupby("Hour")["Usage_kWh"].mean()
plt.bar(hourly.index, hourly.values, color="#3B6E8F")
plt.xlabel("Hour of Day")
plt.ylabel("Average Usage (kWh)")
plt.title("Average Energy Usage by Hour")
plt.tight_layout()
plt.savefig("app/static/energy_by_hour.png", dpi=110)
plt.close()

# 2. Energy usage by load type
plt.figure(figsize=(8,5))
load_type = df.groupby("Load_Type")["Usage_kWh"].mean().sort_values()
plt.barh(load_type.index, load_type.values, color="#5F9EA0")
plt.xlabel("Average Usage (kWh)")
plt.title("Average Energy Usage by Load Type")
plt.tight_layout()
plt.savefig("app/static/energy_by_load_type.png", dpi=110)
plt.close()

# 3. Correlation heatmap (numeric columns only)
plt.figure(figsize=(9,7))
numeric_df = df.select_dtypes(include="number").drop(columns=["HighLoad"], errors="ignore")
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("app/static/correlation_heatmap.png", dpi=110)
plt.close()

print("Saved 3 EDA plots to app/static/")
