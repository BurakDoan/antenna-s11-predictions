import streamlit as st
import joblib
import numpy as np

# 1. Modeli doğrudan yükleyin
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl") # veya models/best_model.pkl

model = load_model()

st.title("📡 Antenna S11 Prediction App")

# Kullanıcı girdileri (Örnektir, kendi parametrelerinizle değiştirin)
L = st.number_input("Patch Length (L)", value=10.0)
W = st.number_input("Patch Width (W)", value=12.0)

if st.button("Predict S11"):
    # 2. Doğrudan model üzerinden tahmin yapın (requests.post YERİNE)
    input_data = np.array([[L, W]])
    prediction = model.predict(input_data)[0]
    
    st.success(f"Tahmin Edilen S11 Değeri: {prediction:.2f} dB")