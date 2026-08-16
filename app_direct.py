import streamlit as st
import joblib
import numpy as np

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Microstrip Antenna S11 Predictor",
    page_icon="📡",
    layout="centered"
)

st.title("📡 Microstrip Antenna S11 Prediction")
st.write("Predict the reflection coefficient ($S_{11}$) directly using Machine Learning.")

# Model Yükleme
@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Kullanıcı Girdileri (5 Temel Anten Parametresi)
st.header("Antenna Parameters")

col1, col2 = st.columns(2)

with col1:
    patch_length = st.number_input("Patch Length (mm)", value=10.0, step=0.1)
    patch_width = st.number_input("Patch Width (mm)", value=10.0, step=0.1)
    ground_length = st.number_input("Ground Length (mm)", value=20.0, step=0.1)

with col2:
    ground_width = st.number_input("Ground Width (mm)", value=20.0, step=0.1)
    gap = st.number_input("Gap (mm)", value=1.0, step=0.1)

# Tahmin Butonu ve Hesaplama
if st.button("Predict S11", use_container_width=True):
    # Modelin beklediği tam sıra: patch_length, patch_width, ground_length, ground_width, gap
    features = np.array([[patch_length, patch_width, ground_length, ground_width, gap]])
    
    try:
        prediction = model.predict(features)[0]
        st.markdown("---")
        st.metric(label="Predicted S11 Parameter", value=f"{prediction:.2f} dB")
    except Exception as e:
        st.error(f"Prediction Error: {e}")