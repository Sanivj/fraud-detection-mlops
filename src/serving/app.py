from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np

BASE_DIR = Path("D:/LearningPython/fraud-detection")

model = joblib.load(BASE_DIR / "models/fraud_model.pkl")
scaler = joblib.load(BASE_DIR / "models/scaler.pkl")

with open(BASE_DIR / "models/metadata.json") as f:
    metadata = json.load(f)

THRESHOLD = metadata["threshold"]
FEATURES = metadata["features"]

app = FastAPI(title="Fraud Detection API", version="1.0.0")

class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount_scaled: float
    Time_scaled: float

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": metadata["model"],
        "threshold": THRESHOLD,
        "roc_auc": metadata["roc_auc"]
    }

@app.post("/predict")
def predict(transaction: Transaction):
    features = np.array([[getattr(transaction, f) for f in FEATURES]])
    prob = model.predict_proba(features)[0][1]
    is_fraud = bool(prob >= THRESHOLD)
    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(float(prob), 4),
        "threshold_used": THRESHOLD,
        "risk_level": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
    }