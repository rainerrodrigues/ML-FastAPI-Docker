from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Sklearn Interence API")

model = joblib.load("app/model.joblib")

class IrisRequest(BaseModel):
	features: list[float] # sepal_length, sepal_width, petal_length, petal_width
	
@app.get("/")
def health():
	return {"status":"ok"}
	
@app.post("/predict")
def predict(request: IrisRequest):
	X = np.array(request.features).reshape(1,-1)
	prediction = model.predict(X)[0]
	return {"prediction": int(prediction)}
