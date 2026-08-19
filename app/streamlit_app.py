import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

# --- HEADER ---
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("### AI-powered clinical decision support tool")

# --- CRITICAL DISCLAIMER ---
st.warning("""
⚠️ **MEDICAL DISCLAIMER**: This tool is for EDUCATIONAL and DEMONSTRATION purposes only. 
It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
Always consult a qualified healthcare provider for medical concerns.
""")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open("models/best_xgb.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_model()
    st.success("✅ Model loaded successfully")
except FileNotFoundError:
    st.error("❌ Model files not found. Please check models/ directory.")
    st.stop()

# --- SIDEBAR: PROJECT INFO ---
with st.sidebar:
    st.header("📊 About")
    st.markdown("""
    This app uses **XGBoost** trained on the Cleveland Heart Disease Dataset (303 patients) 
    to predict heart disease risk from 13 clinical features.
    
    **Model Performance:**
    - Accuracy: ~87%
    - ROC-AUC: ~94%
    
    **Tech Stack:**
    - Python, scikit-learn, XGBoost
    - SHAP for explainability
    - Streamlit for deployment
    
    **[View on GitHub](https://github.com/rishikaevenjalin-bit/heart-disease-prediction-ml)**
    """)
    
    st.divider()
    st.markdown("Made with ❤️ by Rishika Evenjalin")

# --- MAIN FORM ---
st.header("📋 Enter Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 20, 100, 50)
    sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", 
                     options=[1, 2, 3, 4],
                     format_func=lambda x: {
                         1: "Typical angina",
                         2: "Atypical angina",
                         3: "Non-anginal pain",
                         4: "Asymptomatic"
                     }[x])
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 220, 130)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 240)

with col2:
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dL?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG",
                          options=[0, 1, 2],
                          format_func=lambda x: {
                              0: "Normal",
                              1: "ST-T abnormality",
                              2: "Left ventricular hypertrophy"
                          }[x])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.radio("Exercise-Induced Angina?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)

with col3:
    slope = st.selectbox("Slope of Peak Exercise ST",
                        options=[1, 2, 3],
                        format_func=lambda x: {
                            1: "Upsloping",
                            2: "Flat",
                            3: "Downsloping"
                        }[x])
    ca = st.selectbox("Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia",
                       options=[0, 1, 2],
                       format_func=lambda x: {
                           0: "Normal",
                           1: "Fixed defect",
                           2: "Reversible defect"
                       }[x])

# --- PREDICT ---
st.divider()

if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
    # Build input
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                thalach, exang, oldpeak, slope, ca, thal]],
                              columns=["age", "sex", "cp", "trestbps", "chol", "fbs",
                                      "restecg", "thalach", "exang", "oldpeak", "slope",
                                      "ca", "thal"])
    
    # Scale
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Display result
    st.header("📊 Prediction Result")
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        if probability >= 0.7:
            st.error(f"### 🔴 HIGH RISK\n\n**Probability: {probability*100:.1f}%**")
        elif probability >= 0.4:
            st.warning(f"### 🟡 MODERATE RISK\n\n**Probability: {probability*100:.1f}%**")
        else:
            st.success(f"### 🟢 LOW RISK\n\n**Probability: {probability*100:.1f}%**")
    
    with col_b:
        # Progress bar
        st.markdown("**Risk Level:**")
        st.progress(float(probability))
        
        st.markdown(f"""
        **Interpretation:**
        - Predicted class: {"Heart Disease Present" if prediction == 1 else "No Heart Disease"}
        - Confidence: {max(probability, 1-probability)*100:.1f}%
        """)
    
    # Reminder disclaimer
    st.info("💡 **Remember**: This is a predictive tool, not a diagnosis. Please consult a healthcare professional.")

