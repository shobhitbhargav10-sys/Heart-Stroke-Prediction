import streamlit as st
import pandas as pd
import joblib


# --------------------------------
# Load model, scaler and columns
# --------------------------------

model = joblib.load("KNN_Heart.pkl")
scaler = joblib.load("Scaler.pkl")
expected_columns = joblib.load("columns.pkl")


# --------------------------------
# Title
# --------------------------------

st.title("Heart Stroke Prediction by Shobhit!")

st.markdown("Provide the following details")


# --------------------------------
# Input fields
# --------------------------------

Age = st.slider(
    "Age",
    18,
    100,
    40
)

Sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

Chest_pain = st.selectbox(
    "Chest pain type",
    ["ATA", "NAP", "TA", "ASY"]
)

Resting_BP = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=200,
    value=120
)

Cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200
)

Fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)

Resting_ECG = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

Max_hr = st.slider(
    "Max Heart Rate",
    60,
    220,
    150
)

Exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["Y", "N"]
)

Oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    0.0,
    6.0,
    1.0
)

ST_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)


# --------------------------------
# Prediction
# --------------------------------

if st.button("Predict"):

    raw_input = {
        "Age": Age,
        "RestingBp": Resting_BP,
        "Cholesterol": Cholesterol,
        "Fasting_bs": Fasting_bs,
        "MaxHR": Max_hr,
        "Oldpeak": Oldpeak,
        "Sex_" + Sex: 1,
        "ChestPainType_" + Chest_pain: 1,
        "RestingECG_" + Resting_ECG: 1,
        "ExerciseAngina_" + Exercise_angina: 1,
        "ST_Slope_" + ST_slope: 1
    }

    # Convert input into DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns in correct order
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # Display result
    if prediction == 1:
        st.error("High Risk Of Heart Disease")
    else:
        st.success("Low Risk Of Heart Disease")