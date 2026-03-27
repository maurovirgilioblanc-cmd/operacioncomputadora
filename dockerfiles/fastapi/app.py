import os
import numpy as np
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "stroke-predictor"
THRESHOLD = float(os.getenv("STROKE_THRESHOLD", "0.17847"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Stroke Prediction API")

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
        print("Modelo cargado correctamente.")
    except Exception as e:
        print(f"Error cargando modelo: {e}")

class PacienteInput(BaseModel):
    age: float
    hypertension: int
    heart_disease: int
    gender: str  # "Male", "Female"
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/predict")
def predict(paciente: PacienteInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    df = pd.DataFrame([paciente.dict()])

    # Feature engineering igual al notebook
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 50, 70, 120],
                              labels=["Joven", "Adulto", "Mayor", "Anciano"]).astype(str)
    df["avg_glucose_level"] = np.clip(df["avg_glucose_level"], 50, 300)
    df["has_risk_factors"] = ((df["hypertension"] == 1) | (df["heart_disease"] == 1)).astype(int)
    df["is_female"] = (df["gender"] == "Female").astype(int)
    df = df.drop(columns=["gender"])
    df["work_type"] = df["work_type"].replace({"Never_worked": "children"})

    proba = model.predict_proba(df)[:, 1][0]
    prediccion = int(proba >= THRESHOLD)

    if proba < 0.3:
        riesgo = "Bajo"
    elif proba < 0.6:
        riesgo = "Medio"
    else:
        riesgo = "Alto"

    return {
        "probabilidad_acv": round(float(proba), 4),
        "prediccion": prediccion,
        "riesgo": riesgo,
        "threshold_usado": THRESHOLD,
    }