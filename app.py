import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")       # Replace with your model filename
scaler = joblib.load("minmaxscaler.pkl")     # Replace with your scaler filename

# App title
st.title("🔬 Breast Cancer Survival Prediction Model")
st.write("This application predicts if a breast cancer patient will survive (Alive or Dead).")

# Input fields
age = st.number_input("Age at Diagnosis", min_value=0)
grade = st.selectbox("Neoplasm Histologic Grade", [1, 2, 3])
stage = st.selectbox("Tumor Stage", [0, 1, 2, 3, 4])
size = st.number_input("Tumor Size (mm)", min_value=0)
nodes = st.number_input("Lymph Nodes Examined Positive", min_value=0)
mutations = st.number_input("Mutation Count", min_value=0)
npi = st.number_input("Nottingham Prognostic Index", min_value=0.0)

er_status = st.selectbox("ER Status", [0, 1])  # 0 = Negative, 1 = Positive
pr_status = st.selectbox("PR Status", [0, 1])
her2_status = st.selectbox("HER2 Status", [0, 1])

# Predict button
if st.button("Predict Survival Status"):
    # Prepare input for prediction
    input_data = np.array([[age, grade, stage, size, nodes, mutations, npi, er_status, pr_status, her2_status]])

    # Scale input
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # Output
    if prediction == 1:
        st.success("🟢 The patient is likely to survive (Alive).")
    else:
        st.error("🔴 The patient is likely not to survive (Dead).")
