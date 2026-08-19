import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="🫀", layout="wide")

st.title("🫀 Heart Disease Risk Predictor")
st.warning("⚠️ **DISCLAIMER**: For educational purposes only. NOT medical advice.")

@st.cache_resource
def load_model():
    with open("models/best_xgb.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

st.header("📋 Enter Patient Information")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", 20, 100, 50)
    sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
    trestbps = st.number_input("Resting BP", 80, 220, 130)
    chol = st.number_input("Cholesterol", 100, 600, 240)

with col2:
    fbs = st.radio("Fasting BS > 120?", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.radio("Exercise Angina?", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 7.0, 1.0, step=0.1)

with col3:
    slope = st.selectbox("Slope", [1, 2, 3])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [0, 1, 2])

if st.button("🔍 Predict Risk", type="primary"):
    data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                       columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"])
    scaled = scaler.transform(data)
    prob = model.predict_proba(scaled)[0][1]

    if prob >= 0.7:
        st.error(f"🔴 HIGH RISK — {prob*100:.1f}%")
    elif prob >= 0.4:
        st.warning(f"🟡 MODERATE RISK — {prob*100:.1f}%")
    else:
        st.success(f"🟢 LOW RISK — {prob*100:.1f}%")

    st.progress(float(prob))

st.divider()
st.markdown("Made by Rishika Evenjalin | [GitHub](https://github.com/rishikaevenjalin-bit/heart-disease-prediction-ml)")
