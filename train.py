from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

X, y = load_iris(return_X_y=True)


model = RandomForestClassifier(n_estimators=100)
model.fit(X,y)

joblib.dump(model,"app/model.joblib")

print("Model trained and saved.")
