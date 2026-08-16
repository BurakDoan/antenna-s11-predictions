from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Antenna S11 Predictor API", version="1.0")

model = joblib.load("models/best_model.pkl")

class AntennaInput(BaseModel):
    patch_length: float
    patch_width: float
    ground_length: float
    ground_width: float
    gap: float

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Antenna S11 Inference API"}

@app.post("/predict")
def predict_s11(data: AntennaInput):
    try:
        features = np.array([[
            data.patch_length,
            data.patch_width,
            data.ground_length,
            data.ground_width,
            data.gap
        ]])
        prediction = model.predict(features)[0]
        return {"S11_dB_prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))