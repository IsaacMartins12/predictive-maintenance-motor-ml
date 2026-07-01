from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Industrial Motor Risk Predictor"
)


class MotorData(BaseModel):
    voltage: float = Field(..., description="Tensão em Volts (V)")
    current: float = Field(..., description="Corrente em Amperes (A)")
    temperature: float = Field(..., description="Temperatura em °C")
    vibration: float = Field(..., description="Vibração em mm/s")


class PredictionResult(BaseModel):
    risk: str


@app.post("/predict", response_model=PredictionResult)
def predict(data: MotorData):

    sample = pd.DataFrame([{
        "Voltage (V)": data.voltage,
        "Current (A)": data.current,
        "Temperature (°C)": data.temperature,
        "Vibration (mm/s)": data.vibration
    }])

    prediction = model.predict(sample)

    risk_value = prediction[0]

    return PredictionResult(risk=risk_value)
