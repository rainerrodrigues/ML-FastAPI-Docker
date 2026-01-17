import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

params = yaml.safe_load(open("params.yaml"))

df = pd.read_csv("data/processed/train.csv")
X = df.drop("target", axis=1)
y = df["target"]

mlflow.set_experiment("reproducible-iris")

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=params["train"]["n_estimators"],
        max_depth=params["train"]["max_depth"],
        random_state=42
    )

    model.fit(X, y)
    preds = model.predict(X)

    acc = accuracy_score(y, preds)

    mlflow.log_params(params["train"])
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)

