import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load saved model and feature columns
# --------------------------------------------------
model = joblib.load("used_car_price_model.pkl")
feature_columns = joblib.load("model_features.pkl")

# Initialize SHAP explainer (once)
explainer = shap.TreeExplainer(model)

df = pd.read_csv("cardekho_dataset.csv")

brands = sorted(df["brand"].unique())

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")

st.title("Used Car Price Prediction")
st.write("Enter car details to get an estimated selling price.")

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
brands_with_other = brands + ["Other"]
brand_selected = st.selectbox("Brand", brands_with_other)

if brand_selected == "Other":
    brand = st.text_input("Enter Brand Name")
else:
    brand = brand_selected

# Model depends on brand
if brand_selected != "Other":
    models = sorted(df[df["brand"] == brand_selected]["model"].unique())
else:
    models = []

models_with_other = models + ["Other"]
model_selected = st.selectbox("Model", models_with_other)

if model_selected == "Other":
    model_name = st.text_input("Enter Model Name")
else:
    model_name = model_selected

km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
mileage = st.number_input("Mileage (km/l)", min_value=0.0, value=18.0)
engine = st.number_input("Engine Capacity (cc)", min_value=500, value=1200)
max_power = st.number_input("Max Power (bhp)", min_value=20.0, value=80.0)
seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9], index=2)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
transmission_type = st.selectbox("Transmission", ["Manual", "Automatic"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer"])

# SHAP toggle
show_explanation = st.checkbox("Explain prediction using SHAP")

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if st.button("Predict Price"):
    input_dict = {
        "brand": brand,
        "model": model_name,
        "km_driven": km_driven,
        "vehicle_age": vehicle_age,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "fuel_type": fuel_type,
        "transmission_type": transmission_type,
        "seller_type": seller_type
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode
    input_encoded = pd.get_dummies(input_df)

    # Align with training features
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # Predict log price
    log_price_pred = model.predict(input_encoded)[0]

    # Convert back to actual price
    price_pred = np.expm1(log_price_pred)

    st.success(f"Estimated Selling Price: ₹ {int(price_pred):,}")

    # --------------------------------------------------
    # SHAP Explainability
    # --------------------------------------------------
    if show_explanation:
        st.subheader("Prediction Explanation (SHAP)")

        shap_values = explainer.shap_values(input_encoded)

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_encoded.iloc[0],
                feature_names=input_encoded.columns
            ),
            show=False
        )
        st.pyplot(fig)

        st.caption(
            "SHAP explains how each feature increased or decreased the predicted price compared to the average market value."
        )
