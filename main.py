from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
import os
import csv
import io

# ===============================
# App Initialization
# ===============================
app = FastAPI(title="Sybil & Sensor Spoofing Detection API")

# ===============================
# CORS Configuration
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://6000-firebase-studio-1765414312477.cluster-osvg2nzmmzhzqqjio6oojllbg4.cloudworkstations.dev",
        "https://6000-firebase-threatwatch-avdup-1767074125878.cluster-osvg2nzmmzhzqqjio6oojllbg4.cloudworkstations.dev",
        "https://6000-firebase-threatwatch-avdup1-1769254947663.cluster-osvg2nzmmzhzqqjio6oojllbg4.cloudworkstations.dev",
        "https://threat-watch-a-vdup1.vercel.app",
       
    ],
    allow_origin_regex=r"https://.*\.cloudworkstations\.dev",#extra-1
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Input Schema
# ===============================
class InputData(BaseModel):
    features: List[float]

# ===============================
# Feature Definitions
# ===============================
REQUIRED_FEATURES = [
    "x",
    "y",
    "speed",
    "acceleration",
]

SENSOR_REQUIRED_FEATURES = [
    "speed_kmh",
    "acceleration_mps2",
    "lane_deviation",
    "obstacle_distance",
    "traffic_density",
]

ACTION_MAP = {
    0: "Normal Driving",
    1: "Sensor Spoofing Detected",
}

# ===============================
# Helpers-extra 2
# ===============================
def to_py(value):
    """Convert numpy values (e.g. np.int64) to plain Python types so JSON can send them."""
    if hasattr(value, "item"):
        return value.item()
    return value


# ===============================
# Model Loading
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "final_rf.pkl")
SENSOR_MODEL_PATH = os.path.join(BASE_DIR, "sensor_model.pkl")

model = None
sensor_model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Main RF model loaded")

    if os.path.exists(SENSOR_MODEL_PATH):
        with open(SENSOR_MODEL_PATH, "rb") as f:
            sensor_model = pickle.load(f)
        print("✅ Sensor spoofing model loaded")

except Exception as e:
    print("❌ Model loading failed:", repr(e))

print("MODEL EXISTS:", os.path.exists(MODEL_PATH))
print("SENSOR MODEL EXISTS:", os.path.exists(SENSOR_MODEL_PATH))
print("SENSOR MODEL TYPE:", type(sensor_model))

# ===============================
# Health Check
# ===============================
@app.get("/")
def health():
    return {"status": "Backend running"}

# ===============================
# Generic Prediction (JSON)
# ===============================
@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Main model not loaded")

    try:
        X = np.array(data.features, dtype=float).reshape(1, -1)
        pred = int(model.predict(X)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(max(model.predict_proba(X)[0]))

        return {
            "prediction": pred,
            "confidence": confidence,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===============================
# Generic CSV Prediction
# ===============================
@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Main model not loaded")

    try:
        content = await file.read()
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")))

        row = next(reader, None)
        if not row:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        features = []
        missing = []

        for col in REQUIRED_FEATURES:
            if col not in row:
                missing.append(col)
            else:
                features.append(float(row[col]))

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}",
            )

        X = np.array(features).reshape(1, -1)
        pred = int(model.predict(X)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(max(model.predict_proba(X)[0]))

        return {
            "prediction": pred,
            "confidence": confidence,
            "used_features": REQUIRED_FEATURES,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))#extra -3

# ===============================
# SENSOR SPOOFING (JSON)
# ===============================
@app.post("/predict-sensor-json")
def predict_sensor_json(data: InputData):
    if sensor_model is None:
        raise HTTPException(status_code=500, detail="Sensor model not loaded")

    if len(data.features) != len(SENSOR_REQUIRED_FEATURES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(SENSOR_REQUIRED_FEATURES)} features, got {len(data.features)}",
        )

    try:
        X = np.array(data.features, dtype=float).reshape(1, -1)

        # ✅ DO NOT cast to int
        action = sensor_model.predict(X)[0]

        confidence = None
        if hasattr(sensor_model, "predict_proba"):
            confidence = float(max(sensor_model.predict_proba(X)[0]))

        return {
            "action": action,
            "confidence": confidence,
        }
except HTTPException:#
        raise#etra -4
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        


# ===============================
# SENSOR SPOOFING (CSV)
# ===============================
# @app.post("/predict-sensor-csv")
# async def predict_sensor_csv(file: UploadFile = File(...)):
#     if sensor_model is None:
#         raise HTTPException(status_code=500, detail="Sensor model not loaded")

#     try:
#         content = await file.read()
#         reader = csv.DictReader(io.StringIO(content.decode("utf-8")))

#         row = next(reader, None)
#         if not row:
#             raise HTTPException(status_code=400, detail="CSV file is empty")

#         features = []
#         missing = []

#         for col in SENSOR_REQUIRED_FEATURES:
#             if col not in row:
#                 missing.append(col)
#             else:
#                 features.append(float(row[col]))

#         if missing:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Missing required columns: {missing}",
#             )

#         X = np.array(features, dtype=float).reshape(1, -1)
#         pred = int(sensor_model.predict(X)[0])

#         confidence = None
#         if hasattr(sensor_model, "predict_proba"):
#             confidence = float(max(sensor_model.predict_proba(X)[0]))

#         return {
#             "prediction": pred,
#             "action": ACTION_MAP.get(pred, "Unknown"),
#             "confidence": confidence,
#             "used_features": SENSOR_REQUIRED_FEATURES,
#         }

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
@app.post("/predict-sensor-csv")
async def predict_sensor_csv(file: UploadFile = File(...)):
    if sensor_model is None:
        raise HTTPException(status_code=500, detail="Sensor model not loaded")

    try:
        # Read uploaded file
        content = await file.read()
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")))

        # Get first row
        row = next(reader, None)
        if not row:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        # Extract features in required order
        features, missing = [], []
        for col in SENSOR_REQUIRED_FEATURES:
            if col not in row:
                missing.append(col)
            else:
                try:
                    features.append(float(row[col]))
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid numeric value in column '{col}'"
                    )

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}"
            )

        # ✅ Validate feature length
        if len(features) != len(SENSOR_REQUIRED_FEATURES):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(SENSOR_REQUIRED_FEATURES)} features, got {len(features)}"
            )

        # Predict
        X = np.array(features, dtype=float).reshape(1, -1)
        action = sensor_model.predict(X)[0]   # DO NOT cast to int

        confidence = None
        if hasattr(sensor_model, "predict_proba"):
            confidence = float(max(sensor_model.predict_proba(X)[0]))

        return {
            "action": action,
            "confidence": confidence,
            "used_features": SENSOR_REQUIRED_FEATURES,
        }
except HTTPException:#
        raise#extra-5
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# from fastapi import FastAPI, HTTPException, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List
# import numpy as np
# import pickle
# import os
# import csv
# import io

# # -------------------------------
# # App
# # -------------------------------
# app = FastAPI(title="Sybil Attack Detection API")


# # -------------------------------
# # CORS
# # -------------------------------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # -------------------------------
# # Input Schema (manual input)
# # -------------------------------
# class InputData(BaseModel):
#     features: List[float]

# # -------------------------------
# # Load Random Forest
# # -------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "final_rf.pkl")
# SS_MODEL_PATH = os.path.join(BASE_DIR, "sensor_model.pkl")


# model = None
# try:
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
#     print("✅ Model loaded successfully:", type(model))
#     with open(SS_MODEL_PATH, "rb") as f1:
#         sensor_model = pickle.load(f1)
#     print("✅ Model loaded successfully:", type(model))
# except Exception as e:
#     print("❌ Model load failed:", repr(e))

# print("MODEL PATH:", MODEL_PATH) 
# print("MODEL EXISTS:", os.path.exists(MODEL_PATH))

# # -------------------------------
# # Health
# # -------------------------------
# @app.get("/")
# def health():
#     return {"status": "Backend running"}

# # -------------------------------
# # Predict (manual / JSON)
# # -------------------------------
# @app.post("/predict")
# def predict(data: InputData):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")

#     try:
#         X = np.array(data.features, dtype=float).reshape(1, -1)
#         pred = model.predict(X)[0]

#         confidence = None
#         if hasattr(model, "predict_proba"):
#             confidence = float(max(model.predict_proba(X)[0]))

#         return {
#             "prediction": int(pred),
#             "confidence": confidence
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# REQUIRED_FEATURES = [
#    'x','y','speed','acceleration'
# ]
#  SENSOR_REQUIRED_FEATURES = ['speed_kmh','acceleration_mps2','lane_deviation','obstacle_distance','traffic_density',]
# ACTION_MAP = {0: "Normal", 1: "Spoofing"}

# @app.post("/predict-csv")
# async def predict_csv(file: UploadFile = File(...)):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")

#     try:
#         content = await file.read()
#         decoded = content.decode("utf-8")
#         reader = csv.DictReader(io.StringIO(decoded))

#         # Read first data row
#         row = next(reader, None)
#         if row is None:
#             raise ValueError("CSV file is empty")

#         # Extract only required features in correct order
#         features = []
#         missing = []

#         for col in REQUIRED_FEATURES:
#             if col not in row:
#                 missing.append(col)
#             else:
#                 features.append(float(row[col]))

#         if missing:
#             raise ValueError(f"Missing required columns: {missing}")

#         X = np.array(features).reshape(1, -1)
#         pred = model.predict(X)[0]

#         confidence = None
#         if hasattr(model, "predict_proba"):
#             confidence = float(max(model.predict_proba(X)[0]))

#         return {
#             "prediction": int(pred),
#             "confidence": confidence,
#             "used_features": REQUIRED_FEATURES
#         }

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


# @app.post("/predict-sensor-json")
# def predict_sensor_json(data: InputData):
#     if sensor_model is None:
#         raise HTTPException(status_code=500, detail="Sensor model not loaded")

#     X = np.array(data.features, dtype=float).reshape(1, -1)
#     pred = int(sensor_model.predict(X)[0])

#     confidence = None
#     if hasattr(sensor_model, "predict_proba"):
#         confidence = float(max(sensor_model.predict_proba(X)[0]))

#     return {"prediction": pred, "action": ACTION_MAP.get(pred, "Unknown"), "confidence": confidence}

# @app.post("/predict-sensor-csv")
# async def predict_sensor_csv(file: UploadFile = File(...)):
#     if sensor_model is None:
#         raise HTTPException(status_code=500, detail="Sensor model not loaded")

#     content = await file.read()
#     decoded = content.decode("utf-8")
#     reader = csv.DictReader(io.StringIO(decoded))

#     row = next(reader, None)
#     if row is None:
#         raise HTTPException(status_code=400, detail="CSV file is empty")

#     features, missing = [], []
#     for col in SENSOR_REQUIRED_FEATURES:
#         if col not in row:
#             missing.append(col)
#         else:
#             try:
#                 features.append(float(row[col]))
#             except ValueError:
#                 raise HTTPException(status_code=400, detail=f"Invalid numeric value in column '{col}'")

#     if missing:
#         raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

#     X = np.array(features, dtype=float).reshape(1, -1)
#     pred = int(sensor_model.predict(X)[0])

#     confidence = None
#     if hasattr(sensor_model, "predict_proba"):
#         confidence = float(max(sensor_model.predict_proba(X)[0]))

#     return {
#         "prediction": pred,
#         "action": ACTION_MAP.get(pred, "Unknown"),
#         "confidence": confidence,
#         "used_features": SENSOR_REQUIRED_FEATURES,
#     }
