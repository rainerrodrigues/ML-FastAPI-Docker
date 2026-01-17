from feast import FeatureStore
import mlflow
import pandas as pd

store = FeatureStore(repo_path="features")

training_df = store.get_historical_features(
    entity_df=pd.DataFrame({"entity_id": [1, 2, 3]}),
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ],
).to_df()

X = training_df.drop(columns=["entity_id", "event_timestamp"])
# y comes from label source (simplified here)

