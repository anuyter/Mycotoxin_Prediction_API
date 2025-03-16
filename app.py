from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Paths for model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Ensure correct directory
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "new_final_selected_features_after_vif.csv")

# Load or create scaler
if not os.path.exists(MODEL_PATH):
    logger.error(f"❌ Model file not found: {MODEL_PATH}")
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

if not os.path.exists(SCALER_PATH):
    logger.warning(f"⚠️ Scaler file missing. Attempting to create from dataset...")

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)

        if "vomitoxin_ppb" not in df.columns:
            logger.error(f"❌ Column 'vomitoxin_ppb' missing in dataset.")
            raise ValueError(f"Column 'vomitoxin_ppb' not found in dataset.")

        X = df.drop(columns=["vomitoxin_ppb"])
        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, SCALER_PATH)
        logger.info(f"✅ Scaler created and saved: {SCALER_PATH}")
    else:
        logger.error(f"❌ Scaler file and dataset not found!")
        raise FileNotFoundError(f"Scaler file '{SCALER_PATH}' and dataset '{DATA_PATH}' not found.")

# Load trained model and scaler
logger.info("🔄 Loading model and scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
logger.info("✅ Model and scaler loaded successfully!")

# Define input format using Pydantic
class InputData(BaseModel):
    features: list[float]

# Root endpoint
@app.get("/")
def home():
    return {"message": "🚀 Mycotoxin Prediction API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to NumPy array
        input_data = np.array(data.features).reshape(1, -1)

        # Check input size
        expected_features = scaler.n_features_in_
        if input_data.shape[1] != expected_features:
            raise HTTPException(status_code=400, detail=f"Expected {expected_features} features, but got {input_data.shape[1]}.")

        # Scale the input
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)[0]

        return {"predicted_vomitoxin_ppb": round(float(prediction), 4)}

    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
