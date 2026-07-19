import streamlit as st
import pickle
import numpy as np

# Load the saved model and scaler
model = pickle.load(open("models/best_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

st.title("Diabetes Risk Predictor")
st.write("Enter patient details below to predict diabetes risk.")

# Inputs — matching your exact training column order
pregnancies = st.slider("Pregnancies", 0, 17, 1)
glucose = st.slider("Glucose", 0, 200, 100)
blood_pressure = st.slider("Blood Pressure", 0, 130, 70)
skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 850, 80)
bmi = st.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 18, 90, 30)

# These two are internal engineered features, not something a user should manually set —
# since a user is entering real values, these are always 0 (not missing)
insulin_was_missing = 0
skinthickness_was_missing = 0

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                             insulin, bmi, dpf, age,
                             insulin_was_missing, skinthickness_was_missing]])
    
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]
    
    if prediction == 1:
        st.error(f"Result: Likely Diabetic (probability: {probability:.1%})")
    else:
        st.success(f"Result: Likely Not Diabetic (probability: {probability:.1%})")
    
    st.caption("This is a demo model for educational purposes, not a medical diagnostic tool.")