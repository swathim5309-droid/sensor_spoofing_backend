from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
import os
import csv
import io

# -------------------------------
# App
# -------------------------------
app = FastAPI(title="Sybil Attack Detection API")


# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Input Schema (manual input)
# -------------------------------
class InputData(BaseModel):
    features: List[float]

# -------------------------------
# Load Random Forest
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_rf.pkl")
SS_MODEL_PATH = os.path.join(BASE_DIR, "sensor_model.pkl")


model = None
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully:", type(model))
    with open(SS_MODEL_PATH, "rb") as f1:
        sensor_model = pickle.load(f1)
    print("✅ Model loaded successfully:", type(model))
except Exception as e:
    print("❌ Model load failed:", repr(e))

print("MODEL PATH:", MODEL_PATH) 
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))

# -------------------------------
# Health
# -------------------------------
@app.get("/")
def health():
    return {"status": "Backend running"}

-------------------------------
Predict (manual / JSON)
-------------------------------
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        X = np.array(data.features, dtype=float).reshape(1, -1)
        pred = model.predict(X)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(max(model.predict_proba(X)[0]))

        return {
            "prediction": int(pred),
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

REQUIRED_FEATURES = [
   'x','y','speed','acceleration'
]

@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        content = await file.read()
        decoded = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(decoded))

        # Read first data row
        row = next(reader, None)
        if row is None:
            raise ValueError("CSV file is empty")

        # Extract only required features in correct order
        features = []
        missing = []

        for col in REQUIRED_FEATURES:
            if col not in row:
                missing.append(col)
            else:
                features.append(float(row[col]))

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        X = np.array(features).reshape(1, -1)
        pred = model.predict(X)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(max(model.predict_proba(X)[0]))

        return {
            "prediction": int(pred),
            "confidence": confidence,
            "used_features": REQUIRED_FEATURES
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-sensor-json")
def predict_sensor_json(data: InputData):
    if sensor_model is None:
        raise HTTPException(status_code=500, detail="Sensor model not loaded")

    X = np.array(data.features, dtype=float).reshape(1, -1)
    pred = int(sensor_model.predict(X)[0])

    confidence = None
    if hasattr(sensor_model, "predict_proba"):
        confidence = float(max(sensor_model.predict_proba(X)[0]))

    return {"prediction": pred, "action": ACTION_MAP.get(pred, "Unknown"), "confidence": confidence}

@app.post("/predict-sensor-csv")
async def predict_sensor_csv(file: UploadFile = File(...)):
    if sensor_model is None:
        raise HTTPException(status_code=500, detail="Sensor model not loaded")

    content = await file.read()
    decoded = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))

    row = next(reader, None)
    if row is None:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    features, missing = [], []
    for col in SENSOR_REQUIRED_FEATURES:
        if col not in row:
            missing.append(col)
        else:
            try:
                features.append(float(row[col]))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid numeric value in column '{col}'")

    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    X = np.array(features, dtype=float).reshape(1, -1)
    pred = int(sensor_model.predict(X)[0])

    confidence = None
    if hasattr(sensor_model, "predict_proba"):
        confidence = float(max(sensor_model.predict_proba(X)[0]))

    return {
        "prediction": pred,
        "action": ACTION_MAP.get(pred, "Unknown"),
        "confidence": confidence,
        "used_features": SENSOR_REQUIRED_FEATURES,
    }
