import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Antenna S11 Predictor", layout="centered")

st.title("📡 Microstrip Antenna S11 Prediction")
st.write("Predict the reflection coefficient ($S_{11}$) directly using Machine Learning.")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl') # Model dosyanın tam adı

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# User Inputs (Kendi parametrelerine göre düzenle)
st.header("Antenna Parameters")
param1 = st.number_input("Substrate Width (W)", value=10.0)
param2 = st.number_input("Substrate Length (L)", value=10.0)
param3 = st.number_input("Substrate Height (h)", value=1.6)

# Prediction
if st.button("Predict S11"):
    features = np.array([[param1, param2, param3]])
    prediction = model.predict(features)[0]
    st.metric(label="Predicted S11 (dB)", value=f"{prediction:.2f} dB")