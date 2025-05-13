
import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('logistic_model.pkl')

st.title("🧠 Student Mental Health Risk Predictor")

st.markdown("Enter the student’s lifestyle and academic data to assess their mental health risk.")

# Collect user input
age = st.slider("Age", 15, 40, 18)
gender = st.selectbox("Gender", ["Male", "Female"])
academic_performance = st.slider("Academic Performance (e.g. GPA)", 0.0, 5.0, 3.0)
financial_stress = st.selectbox("Financial Stress (0 = No, 1 = Yes)", [0, 1])
family_support = st.slider("Family Support Level (0-10)", 0, 10, 5)
peer_support = st.slider("Peer Support Level (0-10)", 0, 10, 5)
exercise_frequency = st.slider("Exercise Frequency per Week", 0, 7, 3)
hours_of_sleep = st.slider("Average Sleep Hours per Night", 0.0, 12.0, 6.0)
screen_time_hours = st.slider("Daily Screen Time (in hours)", 0.0, 24.0, 6.0)
substance_use = st.selectbox("Substance Use (0 = No, 1 = Yes)", [0, 1])
academic_pressure = st.slider("Academic Pressure Level (0-10)", 0, 10, 5)
social_media_use = st.slider("Social Media Use Level (0-10)", 0, 10, 5)
mental_health_support = st.selectbox("Access to Mental Health Support? (0 = No, 1 = Yes)", [0, 1])
diagnosed_mental_illness = st.selectbox("Previously Diagnosed Mental Illness? (0 = No, 1 = Yes)", [0, 1])

# Encode gender
gender_binary = 1 if gender == "Male" else 0

# Feature array
features = np.array([[age, gender_binary, academic_performance, financial_stress,
                      family_support, peer_support, exercise_frequency, hours_of_sleep,
                      screen_time_hours, substance_use, academic_pressure, social_media_use,
                      mental_health_support, diagnosed_mental_illness]])

# Predict
if st.button("Predict Risk"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ Prediction: High Mental Health Risk")
    else:
        st.success("✅ Prediction: Low Mental Health Risk")
