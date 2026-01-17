from feast import FeatureStore

store = FeatureStore(repo_path="features")

@app.post("/predict")
def predict(entity_id: int):
    features = store.get_online_features(
        features=[
            "iris_features:sepal_length",
            "iris_features:sepal_width",
            "iris_features:petal_length",
            "iris_features:petal_width",
        ],
        entity_rows=[{"entity_id": entity_id}],
    ).to_dict()

    X = [[
        features["sepal_length"][0],
        features["sepal_width"][0],
        features["petal_length"][0],
        features["petal_width"][0],
    ]]

    pred = model.predict(X)[0]
    return {"prediction": int(pred)}

