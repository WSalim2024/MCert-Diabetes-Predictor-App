import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import subprocess
import sys
import time

# --- CONFIG ---
st.set_page_config(page_title="Diabetes Risk Dashboard", page_icon="🩺", layout="wide")

# --- PATHS ---
root_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(root_dir, 'models', 'diabetes_model.pkl')
SCALER_PATH = os.path.join(root_dir, 'models', 'scaler.pkl')
TRAIN_SCRIPT_PATH = os.path.join(root_dir, 'model_train.py')

# --- SIDEBAR ---
st.sidebar.header("Patient Vitals")
pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 6)
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 0, 200, 160)
blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 140, 54)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 0, 100, 20)
insulin = st.sidebar.slider("Insulin Level (mu U/ml)", 0, 900, 100)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 52.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.slider("Age", 0, 120, 38)

st.sidebar.markdown("---")
st.sidebar.header("🔧 Admin Actions")

# Initialize Session State for Errors
if 'train_error' not in st.session_state:
    st.session_state.train_error = None

if st.sidebar.button("🔄 Retrain Model"):
    with st.spinner("Training model..."):
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT_PATH],
            capture_output=True,
            text=True,
            cwd=root_dir
        )
        if result.returncode == 0:
            st.sidebar.success("✅ Training Success!")
            st.session_state.train_error = None
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error("❌ Failed")
            st.session_state.train_error = result.stdout + "\n" + result.stderr

# --- MAIN UI ---
st.title("🩺 Diabetes Risk Assessment Dashboard")

if st.session_state.train_error:
    st.error("🚨 Training Failed! See error details below:")
    st.code(st.session_state.train_error, language="bash")

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    if st.button("Generate Risk Analysis"):
        try:
            # 1. Load Model & Scaler
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)

            # 2. Prepare Input Data
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                                      columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                                               'BMI', 'DiabetesPedigreeFunction', 'Age'])

            # 3. CRITICAL FIX: Scale the data!
            # The model expects values like -1.0 to 1.0, not 85 or 160.
            scaled_input = scaler.transform(input_data)

            # 4. Predict using SCALED data
            prediction = model.predict(scaled_input)
            probability = model.predict_proba(scaled_input)[0][1]

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                if prediction[0] == 1:
                    st.error(f"**Result:** High Risk Detected")
                    st.metric("Probability", f"{probability:.1%}")
                else:
                    st.success(f"**Result:** Low Risk Detected")
                    st.metric("Probability", f"{probability:.1%}")

            with col2:
                st.info("Physician Note:")
                if prediction[0] == 1:
                    st.write(
                        "Patient indicators are consistent with diabetes. Recommended action: Schedule follow-up blood work.")
                else:
                    st.write("Patient vitals are within healthy ranges. Advise routine annual screening.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
else:
    st.warning("⚠️ Model missing. Click **'🔄 Retrain Model'** in the sidebar.")