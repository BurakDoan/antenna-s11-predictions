# 📡 End-to-End Data Science & MLOps Pipeline: Microstrip Antenna S11 Prediction

This repository contains an end-to-end Data Science and MLOps solution designed to predict the reflection coefficient ($S_{11}$) parameter of rectangular microstrip patch antennas using Machine Learning algorithms, RESTful APIs, and interactive web interfaces.

---

## 📌 Project Overview
Microstrip antennas are widely used in wireless communications. Determining the $S_{11}$ resonance parameter typically requires time-consuming High-Frequency Structure Simulator (HFSS) simulations. This project leverages Machine Learning to predict $S_{11}$ values instantly based on physical geometric parameters.

---

## 🛠️ Tech Stack & Tools
- **Language:** Python
- **Machine Learning:** Scikit-Learn (Random Forest / Regression Models)
- **Data Analysis & Processing:** Pandas, NumPy
- **API Framework:** FastAPI
- **Web Interface:** Streamlit
- **Model Tracking & Deployment:** Docker, MLflow (Optional / Pipeline)
- **Version Control:** Git, GitHub

---

## 📁 Project Structure
```text
├── data/                  # Hidden via .gitignore (Raw/Processed Datasets)
├── models/                # Saved ML Models (.pkl)
├── app.py                 # Streamlit Web Application
├── main.py                # FastAPI Backend Service
├── requirements.txt       # Project Dependencies
├── .gitignore             # Git Exclude Rules
└── README.md              # Project Documentation

## 💻 Local Installation & Setup

1. **Clone the repository:**
   git clone https://github.com/BurakDoan/antenna-s11-predictions.git
   cd antenna-s11-predictions

2. **Install requirements:**
   pip install -r requirements.txt
   
3.**Launch the application:**
   streamlit run app.py
