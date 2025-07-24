from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iris-api")

# Try loading the model using joblib, else fallback to pickle
try:
    import joblib
    model = joblib.load("model.pkl")
    loader = "joblib"
except Exception:
    import pickle
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    loader = "pickle"

# Iris class labels (standard scikit-learn order)
class_labels = ["setosa", "versicolor", "virginica"]

# Initialize FastAPI app
app = FastAPI(
    title="🌸 Iris Classifier API",
    description="Predict Iris flower species using a trained scikit-learn model.",
    version="1.0.0"
)

# Request body model
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Response model
class PredictionOutput(BaseModel):
    predicted_class: str
    probability: Optional[float]

@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "Welcome to the Iris Classifier API!",
        "model_loader": loader,
        "model_file": "model.pkl",
        "labels": class_labels,
        "version": "1.0.0"
    }

@app.get("/health", tags=["Health"])
def health_check():
    """Simple health check endpoint."""
    try:
        _ = model.predict([[5.1, 3.5, 1.4, 0.2]])
        return {"status": "ok", "model": "ready"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Model is not responding.")

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(data: IrisInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])
        logger.info(f"Received input: {data.dict()}")

        # Make prediction
        prediction = model.predict(input_df)[0]
        predicted_label = class_labels[int(prediction)]

        # Try to get confidence score
        try:
            probs = model.predict_proba(input_df)[0]
            confidence = round(np.max(probs), 4)
        except Exception:
            confidence = None

        logger.info(f"Prediction: {predicted_label}, Confidence: {confidence}")

        return {
            "predicted_class": predicted_label,
            "probability": confidence
        }

    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.")

