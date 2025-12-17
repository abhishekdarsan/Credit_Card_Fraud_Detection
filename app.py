from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import os

# -----------------------------
# Base directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Load artifacts
# -----------------------------
model = joblib.load(os.path.join(BASE_DIR, "artifacts", "fraud_rf_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "artifacts", "scaler.pkl"))

with open(os.path.join(BASE_DIR, "artifacts", "feature_columns.json"), "r") as f:
    feature_columns = json.load(f)

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="Credit Card Fraud Detection API")

# -----------------------------
# Input schema
# -----------------------------
class TransactionInput(BaseModel):
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
    Amount: float

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_fraud_api(data: TransactionInput):
    df = pd.DataFrame([data.model_dump()])

    # Ensure correct column order
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    prob = model.predict_proba(df_scaled)[0][1]
    prediction = int(prob >= 0.5)

    return {
        "fraud_probability": round(float(prob), 4),
        "fraud_prediction": prediction
    }

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}
