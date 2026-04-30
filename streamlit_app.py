import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Wine Quality Prediction", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load("wine_prediction_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

st.title("🍷 Wine Quality Prediction")
st.write("Use the controls in the sidebar to enter wine physicochemical properties and predict the quality label.")

st.sidebar.header("Input Features")
fixed_acidity = st.sidebar.slider("Fixed Acidity", min_value=1.0, max_value=25.0, value=7.4, step=0.1)
volatile_acidity = st.sidebar.slider("Volatile Acidity", min_value=0.01, max_value=2.0, value=0.7, step=0.01)
citric_acid = st.sidebar.slider("Citric Acid", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", min_value=1.0, max_value=400.0, value=46.0, step=1.0)
density = st.sidebar.slider("Density", min_value=0.9800, max_value=1.0100, value=0.9978, step=0.0001, format="%.4f")
sulphates = st.sidebar.slider("Sulphates", min_value=0.01, max_value=2.0, value=0.56, step=0.01)
alcohol = st.sidebar.slider("Alcohol", min_value=5.0, max_value=15.0, value=9.4, step=0.1)

input_df = pd.DataFrame(
    {
        "fixed acidity": [fixed_acidity],
        "volatile acidity": [volatile_acidity],
        "citric acid": [citric_acid],
        "total sulfur dioxide": [total_sulfur_dioxide],
        "density": [density],
        "sulphates": [sulphates],
        "alcohol": [alcohol],
    }
)

st.subheader("Input values")
st.write(input_df)

if st.button("Predict Wine Quality"):
    try:
        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]
        probabilities = model.predict_proba(scaled)[0]
        proba_df = pd.DataFrame(
            {"quality": model.classes_, "probability": probabilities}
        ).sort_values("probability", ascending=False)

        st.success(f"Predicted wine quality: {prediction}")
        st.write(proba_df)
    except Exception as exc:
        st.error(f"Prediction error: {exc}")
