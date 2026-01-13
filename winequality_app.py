import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("wine_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
## prediction= model.predict(input_data)[0]
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Wine Prediction App", layout="centered")

st.title("🍷 Wine Regression App")
st.write(
    "Predict the quality of wine base on alcohol content."
)


st.sidebar.header("🧮 Input Wine Features")


st.sidebar.subheader("Acidity Levels")
fixed_acidity = st.sidebar.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
citric_acid = st.sidebar.number_input("Citric Acid", 0.0, 2.0, 0.0)

st.sidebar.subheader("Chemical Composition")
residual_sugar = st.sidebar.number_input("Residual Sugar", 0.0, 20.0, 1.9)
chlorides = st.sidebar.number_input("Chlorides", 0.0, 1.0, 0.076)
sulphates = st.sidebar.number_input("Sulphates", 0.0, 2.0, 0.56)

st.sidebar.subheader("Sulfur Dioxide")
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", 0.0, 100.0, 11.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", 0.0, 300.0, 34.0)

st.sidebar.subheader("Other Properties")
density = st.sidebar.number_input("Density", 0.0, 2.0, 0.9978)
pH = st.sidebar.number_input("pH", 0.0, 14.0, 3.51)
alcohol = st.sidebar.number_input("Alcohol", 0.0, 20.0, 9.4)


input_data = pd.DataFrame({
    
        "fixed_acidity": [fixed_acidity], # type: ignore
        "volatile_acidity": [volatile_acidity],  # type: ignore
        "citric_acid": [citric_acid], 
        "residual_sugar": [residual_sugar], 
        "chlorides": [chlorides], 
        "free_sulfur_dioxide": [free_sulfur_dioxide], 
        "total_sulfur_dioxide": [total_sulfur_dioxide], 
        "density": [density], 
        "pH": [pH], 
        "sulphates": [sulphates]
    })

st.subheader(" Input Summary")
st.dataframe(input_data)

# Make sure 'quality' isn't expected
if 'quality' in scaler.feature_names_in_:
    cols = [c for c in scaler.feature_names_in_ if c != 'quality']
    input_data = input_data.reindex(columns=cols)


input_data.columns = input_data.columns.str.replace('_', ' ')

scaled_features = scaler.transform(data)
prediction = model.predict(scaled_features)


prediction = np.random.uniform(0, 10)

st.button("Predict 🍷Wine")
st.success(f"{prediction:.2f}/10")

