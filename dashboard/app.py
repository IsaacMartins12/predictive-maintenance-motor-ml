import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"

model = joblib.load(MODEL_PATH)

st.title(
    "Industrial Motor Risk Predictor"
)

voltage = st.slider(
    "Voltage",
    100,
    500,
    380
)

current = st.slider(
    "Current",
    1,
    50,
    10
)

temperature = st.slider(
    "Temperature",
    20,
    120,
    60
)

vibration = st.slider(
    "Vibration",
    0,
    20,
    5
)

if st.button("Predict"):

    sample = pd.DataFrame([{
        "Voltage (V)": voltage,
        "Current (A)": current,
        "Temperature (°C)": temperature,
        "Vibration (mm/s)": vibration,
    }])

    result = model.predict(sample)

    st.success(
        f"Predicted Risk: {result[0]}"
    )