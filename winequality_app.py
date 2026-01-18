import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("wine_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Wine Quality Prediction App", layout="centered")

st.title("🍷 Wine Quality Classification App")
st.write(
    "Predict the quality of wine based on its attributes."
)


st.sidebar.header("🧮 Input Features")

['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol',]

# --- Numerical inputs ---
fixed_acidity = st.sidebar.slider("Fixed Acidity", min_value = 1.0, max_value = 25.0, value = 7.0, step = 0.1)
volatile_acidity = st.sidebar.slider("Volatile Acidity", min_value = 0.01, max_value = 2.0, value = 0.5, step = 0.01)
citric_acid = st.sidebar.slider("Citric Acidity", min_value = 0.01, max_value = 2.0, value = 0.5, step = 0.01)
residual_sugar = st.sidebar.slider("Residual Sugar", min_value = 0.1, max_value = 30.0, value = 10.5, step = 0.1)
chlorides = st.sidebar.slider("Chlorides", min_value = 0.0, max_value = 2.0, value = 0.5, step = 0.01)
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", min_value = 1.0, max_value = 150.0, value = 10.0, step = 0.1)
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", min_value = 1.0, max_value = 200.0, value = 20.0, step = 0.1)
density = st.sidebar.slider("Density", min_value = 0.01, max_value = 2.0, value = 0.5, step = 0.01)
pH = st.sidebar.slider("pH", min_value = 0.01, max_value = 12.0, value = 2.5, step = 0.01)
sulphates = st.sidebar.slider("Sulphates", min_value = 0.01, max_value = 5.0, value = 0.5, step = 0.01)
alcohol = st.sidebar.slider("Volatile Acidity", min_value = 0.0, max_value = 70.0, value = 10.0, step = 0.1)




input_data = pd.DataFrame(
    {
        'fixed acidity': [fixed_acidity], 
        'volatile acidity': [volatile_acidity], 
        'citric acid': [citric_acid], 
        'total sulfur dioxide': [total_sulfur_dioxide], 
        'density': [density], 
        'sulphates': [sulphates],
        'alcohol': [alcohol]

    }
)



# Scale numeric features
scaled_features = scaler.transform(input_data)


if st.button("Predict Wine Quality"):
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]

    if prediction == "Good":
        st.success(f"Predicted: Wine Quality is Good")
    elif prediction == "Average":
        st.warning(f" Predicted: Wine Quality is Average")
    else: 
        st.warning(f" Predicted: Wine Quality is Poor")
