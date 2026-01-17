import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from pathlib import Path

# Load params
params = yaml.safe_load(open("params.yaml"))

# Ensure output directory exists (CRITICAL)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv("data/raw/iris.csv")

train, test = train_test_split(
    df,
    test_size=params["prepare"]["test_size"],
    random_state=params["prepare"]["random_state"]
)

train.to_csv("data/processed/train.csv", index=False)

