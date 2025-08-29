from typing import Dict, List
from fastapi import FastAPI
from pydantic import BaseModel
import os, pandas as pd

# Optional: load a real pipeline if present
MODEL_PIPELINE = None
MODEL_PATH = os.getenv("MODEL_PATH", "models/pipeline.joblib")
try:
    import joblib
    if os.path.exists(MODEL_PATH):
        MODEL_PIPELINE = joblib.load(MODEL_PATH)
except Exception:
    MODEL_PIPELINE = None

app = FastAPI(title="Churn Prediction API", version="0.1.0")

class PredictRequest(BaseModel):
    rows: List[Dict[str, float]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictRequest):
    df = pd.DataFrame(payload.rows)
    if MODEL_PIPELINE is not None:
        preds = MODEL_PIPELINE.predict(df)
        model_name = "pipeline.joblib"
    else:
        # Fallback: "likely churn" if sum(features) < 0 (arbitrary, just to keep the API responsive)
        preds = (df.fillna(0).sum(axis=1) < 0).astype(int).tolist()
        model_name = "dummy"
    return {"preds": list(map(int, preds)), "model": model_name}
