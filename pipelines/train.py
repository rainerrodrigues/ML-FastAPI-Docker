from kfp.dsl import component

@component(
    base_image="python:3.10",
    packages_to_install=["scikit-learn", "joblib"]
)
def train_model(model_path: str) -> None:
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib

    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

