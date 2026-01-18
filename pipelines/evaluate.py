from kfp.dsl import component

@component(
    base_image="python:3.10",
    packages_to_install=["scikit-learn", "joblib"]
)
def evaluate_model(model_path: str, accuracy: float) -> bool:
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    import joblib

    X, y = load_iris(return_X_y=True, as_frame=True)
    model = joblib.load(model_path)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    print(f"Accuracy: {acc}")
    return acc >= accuracy

