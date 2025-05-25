import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware  # ⬅️ Add this

# Initialize FastAPI app
app = FastAPI()

# ⬇️ Add CORS middleware before any route definitions
origins = [
    "http://localhost:5173",  # ⬅️ Add your React frontend's origin
    "https://view-tube-frontend.vercel.app"  # ⬅️ Replace with your deployed frontend if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or use ["*"] for development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and expected columns on startup
with open('health_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('expected_columns.pkl', 'rb') as f:
    expected_columns = pickle.load(f)

# Define request schema using Pydantic
class HealthInput(BaseModel):
    Age: int
    Gender: Literal["Male", "Female"]
    Height: float
    Weight: float
    ScreenTime: float
    ViewingDistance: float
    DeviceUsed: Literal["Smartphone", "Tablet", "Laptop", "Desktop", "TV"]
    VideoBrightness: float
    AudioLevel: float
    SleepSchedule: float
    Headache: Literal["No", "Minor", "Major"]

# Helper function to prepare data
def prepare_for_prediction(example: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(example, drop_first=True)

    for col in expected_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[expected_columns]
    return X

# Prediction route
@app.post("/predictHealthScore")
def predict_health_score(input_data: HealthInput):
    input_df = pd.DataFrame([input_data.dict()])

    try:
        X = prepare_for_prediction(input_df)
        prediction = model.predict(X)[0]

        if prediction >= 90:
            interpretation = "Outstanding health metrics. Excellent balance of habits!"
        elif prediction >= 80:
            interpretation = "Very good health metrics. Keep up the good work!"
        elif prediction >= 60:
            interpretation = "Decent health metrics. You could improve with better screen and sleep habits."
        elif prediction >= 40:
            interpretation = "Moderate health. Consider adjusting routines for better wellness."
        else:
            interpretation = "Poor health metrics. It is advisable to consult a healthcare professional."

        return {
            "PredictedHealthScore": round(prediction, 1),
            "Interpretation": interpretation
        }

    except Exception as e:
        return {"error": f"Failed to make prediction: {str(e)}"}
