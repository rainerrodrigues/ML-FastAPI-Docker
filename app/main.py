from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Sklearn Interence API")

MODEL_VERSION = os.getenv("MODEL_VERSION","v1")

model = joblib.load("app/model.joblib")

class IrisRequest(BaseModel):
	features: list[float] # sepal_length, sepal_width, petal_length, petal_width
	
@app.get("/health/live")
def liveness():
        return {"status": "alive"}
        
@app.get("/health/ready")
def readiness():
        return {
            "status": "ready",
            "model_version": MODEL_VERSION
	
@app.post("/predict")
def predict(request: IrisRequest):
	X = np.array(request.features).reshape(1,-1)
	prediction = model.predict(X)[0]
	return {
	    "prediction": int(prediction),
	    "model_version": MODEL_VERSIOM
	    }
