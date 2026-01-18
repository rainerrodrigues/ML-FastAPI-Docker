import pandas as pd
import numpy as np

# Load reference data
df = pd.read_csv("monitoring/reference.csv")

# Simulate data drift
df["sepal_length"] += np.random.normal(0.5, 0.2, len(df))
df["petal_width"] *= 1.2

# Save as current data
df.to_csv("monitoring/current.csv", index=False)

print("✅ current.csv created with simulated drift")

